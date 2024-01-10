# Adversarial-Learning
Deep Learning Project - Adversarial Attack

_과제로 수행한 프로젝트로, 틀린 내용이 존재할 수 있습니다._

▶ Adversarial (적대적) Learning이란?
* deceptive input (속임수 입력)을 제공하여 머신 러닝 모델을 속이기. 분류를 속이기 위한 것.
* adversarial example들을 generation 하는 것과 detection 하는 것 둘 다를 가리킴.
* 대표적인 예시 : FSGM (Fast Gradient Sign Method)

▶ **"ResNet을 통해 생성된 Adversarial Sample들을 공격하기"**

(0) import / load image data
```python
import os
import math
import csv
import pickle
from urllib import request
import scipy.stats as st

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0")
```
```python
import gdown
ids = [
    '1XHEWIiTv9Czjn9RJ6IHu_fWXfrteks5l',
    '1gwa_5bTO3dchDlC3WZ3nt_0VvwBkvFfi',
    '1DCUuuy20k-dNbUnCyi0HI9O7qy8hOpE5'
]

outputs = [
    'train.csv',
    'test.csv',
    'img.zip'
]


for i, o in zip(ids, outputs):
    gdown.download(id=i, output=o, quiet=False)  # quiet = False: 다운로드 진행 상황 출력되도록 설정

!unzip -qq "/content/img.zip"  # 압축 해제
```
```python
##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(fname):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        image_id_list.append(row['ImageId'])
        label_ori_list.append(int(row['TrueLabel']) - 1)
        label_tar_list.append(int(row['TargetClass']) - 1)
    gt = pickle.load(request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    return image_id_list, label_ori_list, label_tar_list, gt
```
```python
## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
```
```python
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
ids, origins, targets, gt = load_ground_truth('train.csv')

batch_size = 20
max_iterations = 100
input_path = 'images/'
epochs = int(np.ceil(len(ids) / batch_size))

img_size = 299
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound

resnet = models.resnet50(weights="IMAGENET1K_V1").eval()
vgg = models.vgg16_bn(weights="IMAGENET1K_V1").eval()

for param in resnet.parameters():
    param.requires_grad = False
for param in vgg.parameters():
    param.requires_grad = False

resnet.to(device)
vgg.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

(1) 공격 알고리즘 구현

- MI-FGSM (Momentum Iterative Fast Gradient Sign Method)과 TI-FGSM (Targeted Iterative Fast Gradient Sign Method)은 Adversarial Attack 알고리즘 중에서 성능이 좋다고 알려져 있다. 이 둘의 특성을 혼합하여 공격 알고리즘을 만들고자 했다.

① MI-FGSM 특징 : momentum 적용

  - MI-FGSM은 이전 gradient 방향을 유지하기 위해서 momentum을 사용한다.
  - 이는 신속한 수렴에 도움을 줌.

② TI-FGSM 특징 : 감쇠 적용
	
 - TI-FGSM은 감쇠를 통해서 빠른 수렴을 도와준다.
 - 여기서는 감쇠 계수를 적용해서 momentum 방향으로 업데이트 된 delta에 더해줬다.

```python
# MI-FGSM + TI-FGSM

def mixed_attack(X_ori, delta, labels):
    momentum = 0.9  # MI-FGSM에서 사용되는 momentum 값 -> 과거 gradient 방향을 유지하기 위한 비율을 나타냄
    decay_factor = 1.0 # TI-FGSM에서 사용되는 감쇠 계수 -> momentum에 감쇠를 적용하여 더 빠른 수렴을 도와줌

    for t in range(max_iterations): # 공격 반복 횟수에 대한 루프

        logits = resnet(norm(X_ori + delta)) # 현재의 이미지에 대한 모델의 예측 계산
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels) # cross entropy loss
        loss.backward()
        grad_c = delta.grad.clone()
        delta.grad.zero_()

        delta.data = momentum * delta.data + lr * torch.sign(grad_c)  # momentum 방향으로 delta 업데이트 (MI-FGSM 특징)
        delta.data = delta.data - lr * torch.sign(grad_c) # momentum 방향으로 업데이트한 delta에 감쇠를 적용 (TI-FGSM 특징)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255) # delta 클리핑 (일정 범위를 벗어나지 않도록)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori

    return delta
```
```python
preds_ls = []
labels_ls = []
origin_ls = []

torch.cuda.empty_cache()
for k in tqdm(range(epochs), total=epochs):
    batch_size_cur = min(batch_size, len(ids) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))
    ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]
    labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    # Choose one of the attack methods: ifgsm_attack, mifgsm_attack, difgsm_attack, tifgsm_attack
    delta = mixed_attack(X_ori, delta, labels)

    X_pur = norm(X_ori + delta)
    preds = torch.argmax(vgg(X_pur), dim=1)

    preds_ls.append(preds.cpu().numpy())
    labels_ls.append(labels.cpu().numpy())
    origin_ls.append(ori_idx)
```

(2) 모델 정확도 평가
* accuracy score를 계산한 결과, 0.0이 나왔다.
* 이는 모델이 모든 예측에 틀림을 의미한다.
* 즉, 공격이 모든 이미지에 대해 원래의 label 대신 다른 label로 잘못 분류되었음을 나타낸다.
* 따라서, 0.0의 accuracy score는 해당 공격이 목표로 설정한 label로 모든 이미지를 성공적으로 공경했음을 의미한다.

```python
# evaluation
df = pd.DataFrame({
    'origin': [a for b in origin_ls for a in b],
    'pred': [a for b in preds_ls for a in b],
    'label': [a for b in labels_ls for a in b]
})

accuracy_score(df['label'], df['pred'])

df.to_csv('submission.csv')
```

(3) 모델의 공격 효과 평가
* accuracy score이 0.0이 나왔지만, 이것만으로 공격의 효과가 뛰어나다고 판단하기에는 부족하다고 생각했다. (0.0이 나왔다는 것도 조금 의심스러웠다...)
* 따라서, 모델의 공격 효과에 대해서 평가하기 위해 다음의 두 평가 방법을 사용하였다.
* 
  ① Success Rate (공격 성공률) : 공격이 성공적으로 목표한 label로 이미지를 분류한 비율을 나타낸다. ‘공격 성공 수 / 전체 시도 수’로 계산한다.
  
  ② Fooling Rate (속임수 성공률) : 공격된 이미지가 원본 label과 목표 label 사이에서 얼마나 멀리 떠어져 있는지를 고려한 것이다. ‘틀린 label 수 / 전체 시도 수’로 계산한다.
* 해당 지표들을 계산했을 때, ‘Success Rate: 100.00%, Fooling Rate: 100.00%‘으로 나오는 것을 확인할 수 있었다.

```pythonsuccessful_attacks = len(df[df['pred'] != df['label']])

# 전체 시도 수 계산
total_attempts = len(df)

# Success Rate (공격 성공률) 계산
success_rate = successful_attacks / total_attempts
print(f'Success Rate: {success_rate * 100:.2f}%')

# 속임수 성공 수 계산 (원래 라벨과 목표 라벨이 다른 경우)
fooling_attacks = len(df[df['label'] != df['origin']])

# Fooling Rate (속임수 성공률) 계산
fooling_rate = fooling_attacks / total_attempts
print(f'Fooling Rate: {fooling_rate * 100:.2f}%')
```

(4) 결과 시각화
```python
# visualization
def viz(img_A, img_B, origins, labels, gt, preds):
    for img_a, img_b, origin, label, pred in zip(img_A, img_B, origins, labels, preds):
        img_a = img_a.permute(1, 2, 0)
        img_b = img_b.permute(1, 2, 0)

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3))
        axA.imshow(img_a)
        axA.set_title("True label: " + gt[origin])
        axB.imshow(img_b)
        axB.set_title("Target: " + gt[label])

        result = 'Failed' if pred != label else 'Success'
        caption = f'Pred: {gt[pred]} -> {result}'
        fig.text(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

        plt.show()

viz(X_ori.cpu().detach(), X_pur.cpu().detach(), ori_idx, labels.cpu().numpy(), gt, preds.cpu().numpy())
```
![image](https://github.com/YuSol-Oh/Adversarial-Learning/assets/77186075/446a5e2f-8cab-452b-97b5-209a58f64bd8)
![image](https://github.com/YuSol-Oh/Adversarial-Learning/assets/77186075/7091216f-686b-4814-abda-9ee60709bdbe)
![image](https://github.com/YuSol-Oh/Adversarial-Learning/assets/77186075/231c77c2-e12b-4573-9336-fdccb7b3549c)
![image](https://github.com/YuSol-Oh/Adversarial-Learning/assets/77186075/6cf21987-5cc8-4354-8a93-0abd9e10e875)

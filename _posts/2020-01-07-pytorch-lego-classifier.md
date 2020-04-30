---
layout: post
title:  "[Project] Pytorch로 정확도 87% 이상의 LEGO classifier 만들기"
subtitle:   "making lego classifer"
categories: study
tags: project
comments: true
use_math : true
---

&nbsp;&nbsp;&nbsp;&nbsp;Pytorch 기초에 대한 공부를 마치고 MNIST나 CIFAR10 데이터셋처럼 정제된 데이터셋이 아닌 실제 데이터를 기반으로 classifier를 만들어보기로 결심하였습니다. 적절한 데이터셋을 찾던 중 Kaggle에 올라온 LEGO 이미지 데이터셋이 적절하다고 생각했고 이를 기반으로 **LEGO를 종류별로 분류하는 image classifier**를 만들어보았습니다. 

1) 데이터셋 다운로드  
2) 학습, 시험 데이터셋 분리  
3) 데이터셋 로드하기  
4) 시각화하기  
5) 모델 생성하기  
6) 학습 및 평가  

#### 1) 데이터셋 다운로드 

&nbsp;&nbsp;&nbsp;&nbsp;본 프로젝트는 Kaggle에 올라온 [Images of LEGO Bricks](https://www.kaggle.com/joosthazelzet/lego-brick-images) 데이터를 사용하였습니다. 해당 데이터셋은 50종의 LEGO에 대한 총 4만장의 이미지를 가지고 있습니다. google colab에서 해당 데이터셋을 다음과 같은 코드를 통해 다운받았습니다.

```python
from google.colab import files

! pip install -q kaggle
! files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
!kaggle datasets download -d joosthazelzet/lego-brick-images
!ls
!unzip lego-brick-images.zip
```
- 다음과 같은 과정을 통해 데이터를 로컬에 저장한 후 압축하고 다시 구글 드라이브에 올려주고 압축을 해제하는 번거로운 과정없이 쉽게 데이터셋을 다운받을 수 있었습니다! 상세한 과정은 [캐글과 구글 Colab 연결해주기!](캐글과 구글 Colab 연결해주기!) 블로그를 참고하였습니다. 

- (참고로 데이터셋 zip 파일을 파이썬의 zipfile로 압축 해제하면 상당히 많은 시간이 소요됩니다... 쉘로 압축 해제하는 것을 권장합니다)

#### 2) 학습, 시험 데이터셋 분리(train-test split)

데이터셋의 압축을 풀면 다운받은 디렉터리의 구조는 다음과 같습니다. 

```bash
|-Collada models
|-LEGO brick images v1
|-dataset  # 전체 데이터셋
| |-*.jpg
|-validation.txt  # validation용 파일명이 저장됨
```
- 학습에 사용될 총 이미지는 dataset 디렉터리에 저장되어 있습니다. 저는 이를 pytorch에 쉽게 load시키기 위해 dataset에 저장된 이미지를 train, test 디렉터리로 구분하여 저장하였습니다. test용 이미지 목록은 validation.txt에 저장되어 있습니다. 이후 train, test 각각에 대해 label별로 디렉터리를 생성하고 이미지를 레이블별로 분류해주었습니다. 

```python
import os
import shutil

path = ''
base_path = ''with open(path + 'validation.txt') as f:
  vlist = f.readlines()

test_list = []
for tl in vlist:
  test_list.append(tl.strip())

# print(test_list[:10])

os.makedirs('set/train')
os.makedirs('set/test')

train_path = os.path.join(base_path, 'set', 'train')
test_path = os.path.join(base_path, 'set', 'test')

for file in os.listdir(path):
    print(file)
    if file in test_list:
        shutil.copy(path + file, train_path)
    else:
        shutil.copy(path + file, test_path)

for path in [train_path, test_path]:
  for file in os.listdir(path):
    print(file)
    label = ' '.join(file.split('.')[0].split(' ')[:-1])

    try:
      if label not in os.listdir(path):
        os.mkdir(os.path.join(path, label))
      else:
        pass
      shutil.move(os.path.join(path, file), os.path.join(path, label, file))
    except:
      pass
```
- 위와 같은 과정을 거쳐 데이터셋을 train,test 별로 분리하고 레이블별로 이미지를 분류하여 저장하였습니다. 총 4만장의 데이터를 학습 데이터 32000장, 시험 데이터 8000장으로 분리하였습니다. 

![Cap 2020-03-09 17-19-59-460](https://user-images.githubusercontent.com/35513025/76194797-4260a580-622a-11ea-9af3-a2eea75d26ea.png)

- 저는 학습의 편의성을 고려하여 10종('3001 brick 2x4', '3002 brick 2x3', '3003 brick 2x2', '3004 brick 1x2', '3005 brick 1x1', '3010 brick 1x4', '3020 plate 2x4', '3021 plate 2x3', '3022 Plate 2x2', '3023 Plate 1x2')의 LEGO에 대해서만 학습을 진행하였습니다. 학습 데이터는 6400장, 시험 데이터는 1600장을 사용하였습니다.

#### 3) 데이터셋 로드하기(load dataset)

- torchvision의 datasets 모듈을 사용하면 데이터셋을 쉽게 load시킬 수 있습니다. 디렉터리의 이름이 label로, 해당 디렉터리의 하위 파일들은 같은 label에 속한 데이터로 load되기 때문에 매우 편리합니다. 다음과 같이 디렉터리를 구성하면 됩니다!

```bash
|-3001 brick 2x4  # label 명
| |-image1.jpg    # 해당 label에 속한 데이터
| |-image2.jpg
| |...
|-3002 brick 2x3
| |-image641.jpg
| |-image642.jpg
| |...
...
```
- torchvision의 transforms 모듈을 통해 학습 이미지에 대한 정규화와 augmentation을 진행할 수 있습니다. ***주의할 점은 정규화(Normalization)을 진행할 때 전체 데이터셋에 대해 RGB 값별로 평균과 분산을 미리 계산해줘야 합니다.*** 저는 이 점을 간과하고 일괄적으로 0.5로 평균과 분산을 맞춰 loss가 줄어들지 않는 문제에 마주하였습니다;(무엇이 문제인지 모르고 한참 삽질을 한 후 정규화 문제라는 것을 알게 되었습니다ㅜ)

- 데이터셋의 RGB별 평균과 분산은 다음과 같은 코드를 통해 구할 수 있습니다. 

```python
forcal_train_dataset = torchvision.datasets.ImageFolder(
                        root = path + 'train/',
                        transform=transforms.Compose([
                          transforms.ToTensor(),]))

forcal_train_loader = torch.utils.data.DataLoader(
                        forcal_train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)
pop_mean = []
pop_std0 = []
pop_std1 = []
for i, (data,label) in enumerate(forcal_train_loader, 0):
    # shape (batch_size, 3, height, width)
    numpy_image = data.numpy()
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0) # RGB값의 평균
pop_std0 = np.array(pop_std0).mean(axis=0) # RGB값의 분산
pop_std1 = np.array(pop_std1).mean(axis=0) # 자유도가 1인 RGB값의 분산
```

&nbsp;&nbsp;&nbsp;&nbsp;해당 코드는 [Image normalization in Pytorch](https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7)를 참고하였습니다. 위의 과정을 거쳐 RGB 별로 평균과 분산값을 얻게 되었고 이를 기반으로 데이터셋을 load했습니다. 이번 삽질을 통해 input data 정규화가 얼마나 중요한지 알게 되었습니다..

```python
import torch
import torchvision
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

BATCH_SIZE = 64
EPOCHS = 25
NUM_CLASSES = len(os.listdir(path+ 'train/'))
CLASSES = os.listdir(path + 'train/')

train_dataset = torchvision.datasets.ImageFolder(
                        root = path + 'train/',
                        transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.08660578, 0.08660578, 0.08660578),
                                              (0.17553411, 0.17553411, 0.17553411))]))

train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(
                      root = path + 'test/',
                      transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.08660578, 0.08660578, 0.08660578),
                                              (0.17553411, 0.17553411, 0.17553411)),]))

test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
```

#### 4) 시각화하기(visualization)

사용할 데이터를 torch.utils의 make_grid 메서드를 통해 확인해보았습니다. 배치 사이즈를 64로 설정하여 총 64개의 이미지들이 정규화된 결과를 확인할 수 있습니다. 

![76199153-ad15df00-6232-11ea-9f88-8c01aac3ae6a (Custom)](https://user-images.githubusercontent.com/35513025/79832919-61d01e00-83e5-11ea-8354-e6e958e8ebbe.png)


#### 5) 모델 생성하기

학습을 위해 가장 기본적인 CNN 모델 중 하나인 AlexNet을 사용하였습니다. AlexNet 코드는 torchvison zoo 사이트에 공개된 소스코드를 참고하였습니다. 생성자에는 분류할 LEGO의 label 수인 NUM_CLASSES를 입력하였습니다. 

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

class AlexNet(nn.Module):

  def __init__(self, num_classes=NUM_CLASSES):
    super(AlexNet, self).__init__()                      # (batch size, channel, width, height)
    self.features = nn.Sequential(                       # input : (64, 3, 400, 400)
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  #  (64, 64, 99, 99)
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 64, 49, 49)
        nn.Conv2d(64, 192, kernel_size=5, padding=2),           #  (64, 192, 49, 49)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 192, 24, 24)
        nn.Conv2d(192, 384, kernel_size=3, padding=1),          #  (64, 384, 24, 24)
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),          #  (64, 256, 24, 24)
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),          #  (64, 256, 24, 24)
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                  #  (64, 256, 11, 11)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                 #  (64, 256, 6, 6)
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),                           #  (64, 4096)
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),                                  #  (4096, 4096)
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),                           #  (4096, 10)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return F.log_softmax(x, dim=1)
```

- torchvison zoo의 모델을 그대로 사용하자 사이즈가 맞지 않는다는 에러가 발생했습니다. 이를 해결하기 위해 모델을 작성하면서 입력되는 데이터의 (batch size, channel, width, height) 크기를 직접 계산하여 적절한 데이터 사이즈를 입력하여 문제를 해결하였습니다. 

- 이러한 과정에서 Adaptive Average Pooling에 대해 알게 되었습니다. 기존 average pooling과는 다르게 입력으로 받은 정수의 크기로 출력되는 이미지의 크기를 맞춰주는 방식입니다. 

처음에는 최적화 함수를 SGD(Stochastic Gradient Descent)로 설정해주었으나 이로 인해 학습 속도가 지나치게 느린 문제가 발생하였습니다. 이를 해결하기 위해 최적화 함수를 Adam으로 설정해주자 위와 같은 문제를 해결할 수 있었습니다.

```python
model = AlexNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### 6) 학습 및 평가

학습 및 평가를 위한 코드는 다음과 같이 작성하였습니다. 

```python
def train(model, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 20 == 0:
      print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset), 
          100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(DEVICE), target.to(DEVICE)
      output = model(data)

      test_loss += F.cross_entropy(output, target,
                                   reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100 * correct / len(test_loader.dataset)
  return test_loss, test_accuracy
  
for epoch in range(1, EPOCHS+1):
  train(model, train_loader, optimizer, epoch)
  test_loss, test_accuracy = evaluate(model, test_loader)

  print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(
      epoch, test_loss, test_accuracy))
```

- google colab의 GPU를 통해 CUDA를 활용하여 상대적으로 빠르게 학습을 진행할 수 있었습니다. 다만 실행하는 시점에 따라 병목현상이 발생하거나 cuda Error 메시지가 출력될 때가 있어 런타임을 초기화해야하는 번거로움이 있었습니다. google colab Pro을 사용하면 훨씬 더 빠른 속도를 경험할 수 있다하여 바로 결제를 하려 했으나... 아직은 미국에서만 서비스가 가능하다고 하더라구요ㅠㅠ

![Cap 2020-03-14 21-35-58-182](https://user-images.githubusercontent.com/35513025/76682036-dde38300-663b-11ea-9bd0-af4982cdd102.png)

- 학습 결과 최대 87.31%의 정확도를 보였습니다. 성능 개선을 위해 하이퍼 파라미터나 모델을 손보고 있습니다. 

#### Feedback

- MNIST 데이터셋이 얼마나 사용하기 편하게 정리되어 있는지 알게 되었습니다. 모델을 구현하는 부분보다는 데이터를 load시키고 정제하는 과정이 훨씬 더 번거로웠습니다. 이제는 raw dataset을 봐도 train, test 별로 구분하고 label별로 구분하는 작업을 더 잘 할 수 있을 것 같습니다.
 
- 학습 데이터에 대한 정규화(Normalization)와 적절한 optimizer 선택이 모델의 학습 속도와 성능에 미치는 영향을 체감하였습니다. 

- layer을 거듭할수록 변하는 data size를 알아둬야 할 필요성을 체감하였습니다. 주석으로 (batch size, channel, width, height)를 기록하는 습관을 들여야할 것 같습니다.

#### Reference
[캐글과 구글 Colab 연결해주기!](캐글과 구글 Colab 연결해주기!)  
[Image normalization in Pytorch](https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7)  
[3분 pytorch](https://github.com/keon/3-min-pytorch)  

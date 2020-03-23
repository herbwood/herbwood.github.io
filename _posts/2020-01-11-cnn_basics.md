---
layout: post
title:  "[DL] CNN 기본 모델(AlexNet, GoogLeNet, VGG)"
subtitle:   "cnn basic"
categories: study
tags: dl
comments: true
use_math : true
---

  이번 글에서는 image classification 분야에서 활용되는 CNN 모델들에 대해 살펴보도록 하겠습니다. 발전이 빠른 딥러닝 분야에서 5년 이상 된 오래된 모델들은 실용적이지는 않다고 생각합니다. 하지만 모델의 발전 과정을 살펴봄으로써 현재 활용되고 있는 모델에 대한 이해를 도모하고, 개선할 점을 파악할 수 있다고 생각합니다. 따라서 각 모델의 구조를 살펴보기보다는 각 모델이 이미지 인식 분야에 기여한 점을 중점으로 살펴보고자 합니다. 
  

#### AlexNet - CNN의 표준

AlexNet은 CNN 모델 중 최초로 유의미한 성과를 낸 모델로, 이후 모델에도 활용되는 기본 구조를 확립하였습니다. 

![alexnet](https://i.imgur.com/CwIvlUW.png)
- **Multi GPU 사용**
  AlexNet에서는 GPU 2개를 활용하기 위해 모델의 구조를 크게 바꿨습니다. Layer에 따라 GPU가 병렬처리 하는 방식을 구분합니다. AlexNet이 좋은 성과를 낸 이후 연구자들은 모델 학습에 있어 다수의 GPU를 활용하게 됩니다.
 
- **ReLU**
  이전 모델이 활성화 함수로 tanh를 사용했던 것과 달리 ReLU 함수를 처음으로 사용하였습니다. 이를 통해 학습 속도가 빨라졌으며 vanishing gradient 문제도 어느 정도 해소되었습니다. 
  
- **Dropout**
  Dropout을 처음으로 활용하여 overfitting을 효과적으로 방지하였습니다. Dropout은 학습 시에만 적용하고 이후 테스트시에는 모든 뉴런을 사용합니다. 


#### VGG - 적절한 filter size 지정

VGG는 굉장히 일반적인 구조를 가진 모델이지만 신경망이 깊어질수록 성능이 좋아진다는 점을 밝혀냈으며 filter size의 적절한 선택이 가지는 이점을 잘 보여주었습니다. 

![vgg](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network.jpg)

- **3x3 filter의 적절한 활용**
  VGG는 filter size를 3x3으로 고정함으로써 학습시켜야할 파라미터 수를 크게 줄이고 신경망 층의 수를 늘렸습니다. 7x7 filter를 한 번 사용하는 것보다 3x3 filter를 3번 사용하는 것이 파라미터 수가 더 적습니다. 이를 활용하여 신경망의 층을 더 깊게 설계하는 것이 가능해집니다. 

#### GoogLeNet - 더 넓고 깊어진 모델

GoogLeNet은 신경망을 단순히 깊게 쌓은 것이 아니라 단일한 입력값에 대한 서로 다른 연산 결과를 취합하는 모듈을 활용하였으며 1x1 filter를 활용하여 신경망을 이전보다 더 깊게 쌓았습니다. 

![googlenet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FMzPze%2FbtqyQy5e3NM%2F5HPtmAwVQzKJTj6wgWautk%2Fimg.png)

- **Inception module**
  Inception module은 입력값에 대해 서로 다른 filter size를 적용해줌으로써 보다 단일 입력값에 대한 보다 풍부한 특성을 추출합니다. 또한 1x1 filter를 사용하여 입력 데이터의 높이와 넓이는 유지한 채로 channel, 즉 feature map의 수를 효과적으로 줄였습니다. 이를 통해 신경망을 더 깊게 쌓는 것이 가능해졌습니다. 
  
  
#### Reference
[cs231n youtube 강의](https://www.youtube.com/playlist?list=PLzUTmXVwsnXod6WNdg57Yc3zFx_f-RYsq)  
[CNN 알고리즘들](https://bskyvision.com/539?category=635506)  
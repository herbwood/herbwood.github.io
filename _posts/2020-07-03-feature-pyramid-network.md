---
layout: post
title:  "[DL]  Feature Pyramid Network 논문 리뷰"
subtitle:   "feature pyramid network"
categories: study
tags: dl
comments: true
use_math : true
---

이번 포스팅에서는 Feature Pyramid Network에 대해 살펴보도록 하겠습니다. 사실 EfficientDet 논문을 읽어보면서 FPN와 관련된 내용들이 있어 먼저 읽어봐야겠다고 생각했습니다. 실제로 FPN이 많은 Object Detection 모델에 영향을 끼친만큼 공부할 필요가 있을 것 같습니다. 

## What's the Problem?

Object Detection 알고리즘은 작은 객체에 대해서 제대로 탐지를 하지 못한다는 문제가 있습니다. 이를 해결하기 위해 이미지의 scale을 변화시켜가면서 객체를 탐지하는 방법들이 등장했습니다. 입력값의 scale을 변화시켜준 feature map을 차례로 쌓기 때문에 feature pyramid라고 부르는 것 같습니다.  아래의 그림은 이미지와 feature map의 크기를 변화시켜 객체 탐지에 사용한 기법들의 예시입니다(자세한 설명은 [갈아먹는 머신러닝의 블로그](https://yeomko.tistory.com/44?category=888201)님를 참고했습니다).

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc5D2i4%2FbtqEfqUaK1s%2Fk5kgInuWo7qu1ik0IP6Tz1%2Fimg.png" width="600px"></p><p align="center">[그림 1] 다양한 방식의 feature pyramid</p>

(a) 방식은 이미지의 입력 scale을 다양하게 만들어 각각의 이미지에 대한 feature map을 만든 후 객체를 탐지하는 방식입니다. 입력 이미지를 복사한 후 여러 번 resize해줘야 하기 때문에 연산량이 많아 속도가 느리다는 단점이 있습니다.   

(b) 방식은 입력 이미지를 CNN 네트워크에 넣은 후 마지막 layer에서 얻은 feature map을 통해 객체를 탐지하는 기법입니다. layer를 거치면서 입력값의 scale이 계속 작아지면서 작은 객체를 탐지하기 어려운 문제가 있습니다. 

(c) 방식은  layer를 거치면서 생성되는 각각의 featue map을 활용하여 객체를 탐지하는 기법입니다. [SSD](https://herbwood.github.io/study/2020/04/18/ssd/)가 이에 해당합니다. SSD는 VGG를 baseline 모델로 사용하여 6단계의 layer에서 도출되는 feature map을 사용하여 각 featue map에서 객체를 탐지합니다. 하지만 상위 layer의 추상화된 정보를 활용하지 못한다는 문제(large semantic gaps caused by different depth)가 있습니다. 

논문의 저자는 위의 세 가지 모델에서 나타나는 문제를 해결하기 위해 (d) 형태의 새로운 구조의 모델을 제안합니다. 추상적인 의미를 많이 담고 있는(semantically strong) 저해상도(low-resolution) feature map과 의미론적으로 약한(semantically weak) 고해상도(high resolution) feature map을 연결하는 **Feature Pyramid Network**를 고안합니다. 

<br>

## Improvements

Feature Pyramid Network는 임의의 크기의 이미지를 입력 받아 multi-level에 걸쳐 비례하는 크기의 feature map을 산출하는 방식입니다. Backbone CNN 모델은 크게 상관없다고 하며, 논문에서는 ResNet을 사용했다고 합니다. Pyramid는 buttom-up pathway, top-down pathway, lateral connections 방식으로 구성됩니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/40735375/71475054-e50ba980-2821-11ea-8a6f-3173c180afbd.png" width="600px"></p><p align="center">[그림 2] Buttom-up, Top-down, lateral connections</p>


### Bottom-up pathway

- **Bottom-up pathway**는 backbone CNN 네트워크에서 순전파를 진행하는 과정입니다. 이 과정에서 feature map의 해상도는 layer를 거침에 따라 2배씩 작아집니다. 간혹 같은 크기의 feature map을 산출하는 layer가 있는데 그 같은 경우 같은 stage에 있다고 가정합니다.

-  각 stage를 하나의 pyramid level로 정의합니다. 각 stage의 마지막 layer의 output을 feature pyramid를 쌓을 때 참고하는 feature map으로 사용합니다. 이러한 선택은 각 stage별로 가장 깊은 layer는 가장 강력한 feature를 담고 있을 것이기 때문에 충분히 합리적입니다. ResNet에서 C2, C3, C4, C5를 conv2,  conv3, conv4, conv5의 산출물로 봅니다. 

### Top-down pathway and lateral connections

**Top-down pathway**와 **lateral connections** 단계에서는 

1) pyramid에서 높은 stage에 해당하는 고해상도 feature map을 upsampling합니다(논문의 저자는 nearest neighbor upsampling을 사용했다고 합니다) 

2) 아래 stage feature map에 대해서는 1x1 convolution을 적용하여 채널 수를 줄여줍니다. 

3) 그리고 서로 맞닿은 feature map끼리 element-wise하게 더해줍니다. 

4) 이 과정을 최적의 resolution map이 생성될 때까지 반복해줍니다. 

5) 마지막 feature map에 대해서는 3x3 conv를 적용합니다. 

위의 과정을 통해 고해상도 feature map에서 얻을 수 있는 위치에 대한 정보와 저해상도 feature map에서 얻을 수 있는 추상적인 정보를 모두 활용하는 것이 가능해집니다. 

<br>

## Applications

논문의 저자는 Feature Pyramid Network가 다른 Object Detection 알고리즘에 적용하여 좋은 성능을 내는지 확인해보고자 했습니다. 논문에서는 [Faster R-CNN](https://herbwood.github.io/study/2020/03/27/faster-r-cnn/)과 [Fast R-CNN](https://herbwood.github.io/study/2020/03/16/fast-rcnn/) 알고리즘을 통해 성능을 실험했습니다. 

### Faster R-CNN

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FDFL4S%2FbtqEeX5IAp0%2FCbvO9zsvHU9Z6fNcrFkf8K%2Fimg.jpg" width="600px"></p><p align="center">[그림 3] RPN에 FPN을 적용하는 경우</p>

Faster R-CNN 모델에서 RPN(Region Proposal Network)는 single-scale feature map에 대하여 anchor를 적용하여 객체 여부와 bounding box의 offset을 파악합니다. 논문의 저자는 single-scale feature map을 FPN으로 대체합니다. 

- RPN에서 3x3 conv와 1x1 conv를 거쳐 feature map을 반환하는 것과 마찬가지로 FPN은 각 feature pyramid level마다 head(box regressor, classfier)를 적용해줍니다. 

- 또한 multi-scale feature map에 대해 다양한 크기의 anchor를 적용하는 것은 불필요하기 때문에 anchor의 크기를 각 level마다 하나로 고정했습니다. anchor는 5개의 pyramid level마다 하나의 크기를 가지며, aspect ratio는 1:2, 1:1, 2:1을 가져, 총 15개의 anchor를 사용합니다. 

- IOU에 따라 객체 여부를 label하는 방식(IOU가 0.7보다 크면 positive, 0.3보다 작으면 negative)은 변경하지 않았습니다. 

실험을 통해 각 RPN classfier head 부분의 파라미터가 전체 feature pyramid level에 걸쳐 공유되는 것을 확인할 수 있엇습니다. 이것의 장점은 각 level이 비슷한 수준의 추상적 의미(semantically strong)를 가짐을 뜻합니다. 


### Fast R-CNN

Fast R-CNN에서 RoI(Region of Interest)는 single-scale feature map을 사용합니다. FPN을 적용하기 위해서는 RoI를 서로 다른 scale의 pyramid level에 할당해야 합니다. 할당하는 pyramid level은 아래와 같은 공식을 통해 구합니다. 
<p align="center"><img src="https://ifh.cc/g/yiXaVw.png" width="300px"></p>

- 여기서 k는 pyramid level의 인덱스를 의미하며, k0는 target level을 의미합니다. 논문에서 k0=4로 지정했습니다. 직관적으로 봤을 때 RoI의 scale이 작아질수록 낮은 pyramid level에 할당하는 위의 공식은 합리적으로 보입니다. 

- 모든 level의 RoI에 head(box regressor, classifier)를 붙이면 head의 파라미터가 전체 level에 걸쳐 공유되는 것을 확인할 수 있습니다. 

<br>

## Experiments

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlfFW0%2FbtqEgP6Olj2%2Fy1mM8hfKFCA0dCSGKSLDHk%2Fimg.png">[그림 4] 다른 모델과 성능 비교 결과</p>


- RPN에 FPN을 적용했을 시 AP 수치가 8% 이상 올랐으며, 작은 객체를 더 잘 탐지하는 결과를 보였습니다. 

- Top-down pathway를 적용하지 않으면 상대적으로 낮은 accuracy를 보입니다. 그 이유는 bottom-up 구조만 적용하게 됨으로써 서로 다른 level간의 semantic gap이 너무 커지기 때문입니다. 

- Top-down pathway만 적용하게 되면 upsampling 효과가 많이 적용되어 sematic한 정보를 많이 포함하지만 고해상도 feature map에서 얻을 수 있는  localization에 대한 정보가 부족합니다. 1x1 lateral connection을 통해 Bottom-up 구조를 통해 localization 정보를 추가할 수 있습니다. 

- pyramid 구조를 사용하지 않고 가장 마지막 feature map인 P2만 사용할 경우 다양한 scale을 포함하지 않기 때문에 anchor 수를 늘려야 합니다. 하지만 실험 결과 anchor 수가 늘어나도 accuracy가 높아지지 않는 모습를 보였습니다. 

<br>

## Conclusion

저는 개인적으로 Feature map의 scale에 따라 담고 있는 정보가 달라진다는 점이 흥미로웠습니다. 논문에서는 low resolution, 즉 convolution 연산과 max pooling이 여러 번 적용되어 크기가 작은 feature map은 semantically strong하다고 여러 번 언급합니다. 처음에는 "semantic"하다는 단어가 무슨 의미로 사용되는지 잘 몰랐지만 여러 블로그를 읽어본 결과, class에 대한 정보를 의미한다는 것을 알게 되었습니다. 반대로 high resolution feature map의 경우, class에 대한 정보는 부족하지만 localization, 즉 class의 위치에 대한 정보를 많이 포함하고 있습니다. feature map의 scale에 따라 담고 있는 서로 다른 정보를 합쳐 더 좋은 성능을 보인다는 점은 직관적으로도 잘 받아들일 수 있었습니다. 

<br>

## Reference

[FPN 논문](https://arxiv.org/pdf/1612.03144.pdf)    
[항상 많이 참고하는 갈아먹는 머신러닝님의 블로그](https://yeomko.tistory.com/44?category=888201)  
[논문을 잘 정리해주신 해주님의 블로그](https://velog.io/@haejoo/Feature-Pyramid-Networks-for-Object-Detection-%EB%85%BC%EB%AC%B8-%EC%A0%95%EB%A6%AC)    
[각 단계별로 그림을 통해 상세하게 잘 설명해주신 잡동사니님의 블로그](https://blog.naver.com/PostView.nhn?blogId=jinyuri303&logNo=221865008339&parentCategoryNo=36&categoryNo=38&viewDate=&isShowPopularPosts=false&from=postView)  

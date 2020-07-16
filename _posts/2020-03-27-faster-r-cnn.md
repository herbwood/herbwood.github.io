---
layout: post
title:  "[DL] Faster R-CNN 논문 리뷰"
subtitle:   "faster r-cnn"
categories: study
tags: dl
comments: true
use_math : true
---

Fast R-CNN은 ROI Pooling layer, multi-task loss 사용으로 single-stage detector 모델로서의 모습을 갖췄지만 여전히 region proposal 작업을 네트워크와 별도로 수행합니다. 이번 포스팅에서는 이러한 문제를 해결하고 진정한 single stage detector의 면모를 갖춘 Faster R-CNN 모델에 대해 살펴보도록 하겠습니다. 

### What's the Problem?

지금까지 region proposal 방식은 selective search를 통해 cpu 내에서 수행되었습니다. 문제는 gpu를 사용할 수 없기에 후보 영역 추천은 느릴 수 밖에 없습니다. 실제로 학습이나 inference 시 대부분의 시간은 selective search에서 소요한다고 합니다. 뿐만 아니라 selective search와 네트워크를 통한 학습이 별도의 과정으로 수행되어 일종의 병목 현상이 발생하며 end-to-end 학습이 불가능합니다. 

### Improvements

#### RPN(Region Proposal Network)

Faster R-CNN은 **RPN(Region Proposal Network)**를 도입하여 이러한 문제를 해결합니다. RPN은 region propsal 과정을 수행하는 네트워크로, 학습 과정 자체에 region proposal 작업을 포함시켜버립니다. 이를 통해 수행 속도가 크게 감소하고 성능 역시 향상되었습니다. Faster R-CNN은 RPN을 통해 진정한 의미의 sigle-stage detector로 거듭나게 됩니다.(이는 SSD, YOLO 등과 같은 one-stage 모델과는 구분됩니다. Faster R-CNN은 여전히 two-stage detector입니다) 

<p align='center'><img src='http://t2m.kr/Yaot0' width='400px'></img></p><p align='center'>[그림 1] Region Proposal Network</p>

RPN의 학습 과정을 살펴보기에 앞서 Faster R-CNN 모델에서 도입한 **Anchor box**라는 개념에 대해 알고갈 필요가 있습니다. Anchor box는 특정 넓이와 높이, 가로세로 비율을 가지고 있는 사전에 정의된 bounding box입니다. convolution 연산 시 filter의 centroid를 기준으로 높이와 넓이, 그리고 각 비율이 다른 서로 다른 Anchor box가 적용되어 box 내에 객체의 존재 여부를 판별합니다. 각 centroid마다 Anchor box를 사용하는 이유는 몇몇 객체를 탐지하는데 있어 단일한 크기의 정사각형보다 다양한 크기의 직사각형 box가 보다 더 적합하기 때문입니다. 가령 서있는 사람을 탐지하고자 할 경우 bounding box가 세로로 긴 형태일 때 목표 대상을 보다 잘 포착할 수 있습니다. 

<p align='center'><img src='https://whal.eu/i/opqd8NKE'></img></p><p align='center'>[그림 2] how Region Proposal Network works </p>

다음으로 RPN의 학습과정을 살펴보도록 하겠습니다.  

1) CNN을 통해 뽑아낸 feature map(H x W x C)을 입력으로 받습니다.  

2) feature map에 3x3 convolution 연산을 256 혹은 512 channel만큼 수행합니다. 이 때, padding을 1로 설정해주어 H x W가 보존될 수 있도록 해줍니다. 수행 결과로 (H x W x 256) 혹은 (H x W x 512) 크기의 두 번째 feature map을 얻습니다.   

3) 먼저 Object 여부를 확인하는 분류 작업을 수행하기 위해서 1 x 1 filter를 적용하여 (2(Object 여부를 나타내는 지표 수) x 9(anchor box 개수)) channel 수 만큼 수행해주며, 그 결과로 H x W x 18 크기의 피쳐맵을 얻습니다. 이제 이 값들을 적절히 reshape 해준 다음 Softmax를 적용하여 해당 anchor box가 Object일 확률 값을 얻습니다.   

4) 두 번째로 Bounding Box Regression 예측 값을 얻기 위한 1 x 1 컨볼루션을 (4 x 9) channel 수 만큼 수행합니다.  

5) 이제 앞서 얻은 값들로 RoI를 계산합니다. 먼저 Classification을 통해서 얻은 물체일 확률 값들을 정렬한 다음, 높은 순으로 K개의 앵커만 추려냅니다. 그 다음 K개의 앵커들에 각각 Bounding box regression을 적용해줍니다. 그 다음 Non-Maximum-Suppression을 적용하여 최적의 RoI(Region of Interest)을 구해줍니다.   

### Model Architecture
<p align='center'><img src='https://www.dropbox.com/s/db46mw6e5gg63m1/Screenshot%202018-05-09%2016.38.34.png?raw=1' width='400px'></img></p><p align='center'>[그림 3] Faster R-CNN 학습 과정 </p>

논문에서는 4-step alternating training 방식을 사용하여 Faster R-CNN을 학습시켰습니다.  

1) RPN을 multi task loss를 사용하여 학습시킵니다.

2) RPN을 통해 얻은 RoI를 활용하여 Fast R-CNN 모델을 학습시킵니다.

3) 앞서 학습시킨 Fast RCNN과 RPN을 불러온 다음, 다른 가중치값들은 고정하고 RPN에 해당하는 레이어들만 fine tune 시킵니다(여기서부터 RPN과 Fast RCNN이 컨볼루션 가중치값을 공유하게 됩니다).

4) 이번에는 RPN의 가중치값은 고정시킨 채, Fast R-CNN에 해당하는 가중치값만 fine tune 시킨다.

### Performance
<p align='center'><img src='https://whal.eu/i/1EwmobM7'></img></p><p align='center'>[그림 3] Faster R-CNN 속도 측정 결과</p>

RPN을 적용한 Faster R-CNN이 다른 Selective Search를 사용한 다른 모델을 속도 측면에서 크게 앞질렀습니다. 초당 17프레임 정도 나오는 결과를 보입니다. 그 밖에 성능 측면에서도 RPN을 적용했을 때 mAP가 소폭 상승하는 결과를 보였습니다. 

### Conclusion
지금까지 Faster R-CNN에 대해 살펴보았습니다. 논문에서 Anchor box, RPN 등 새로운 개념들이 등장하여 이해하기 조금 어려웠던 것 같습니다. 그나마 RPN 외의 나머지 학습과정이 Fast R-CNN과 동일해서 정말 다행이었습니다; Faster R-CNN은 two-stage detctor 중에서 가장 좋은 성능과 속도를 보였지만 여전히 실시간으로 객체를 탐지하는데 약한 모습을 보입니다. 다음 포스팅에서는 실시각 객체 탐지 task에서 좋은 모습을 보인 SSD 모델에 대해 살펴보도록 하겠습니다. 


### Reference
[Faster R-CNN 논문](https://arxiv.org/pdf/1506.01497.pdf)  
[RPN을 이해하는 데 큰 도움이 된 블로그](https://yeomko.tistory.com/17)  
[Faster R-CNN에 대해 설명한 블로그](https://curt-park.github.io/2017-03-17/faster-rcnn/)  
[딥러닝논문읽기모임 유튜브 채널](https://www.youtube.com/watch?v=HmJWvwIpW5g&t=1545s)  
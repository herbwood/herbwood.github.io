---
layout: post
title:  "[DL] Object detection Metric"
subtitle:   "object detection"
categories: study
tags: dl
comments: true
use_math : true
---

최근 컴퓨터 비전 분야에서는 이미지를 분류하는 것을 넘어 이미지의 객체를 탐지하는 Object detection에 대한 연구가 활발히 진행되고 있습니다. Object detection은 자율 주행차, 얼굴 및 보행자 검출, 영상 복구 등 다양한 분야에서 활용되고 있습니다. 저도 Object detection에 관심을 가져 관련 논문을 읽어보고자 했으나, 이미지 분류 분야와는 다른 문제 정의, 해결 방법, 그리고 평가 방식의 차이를 이해하지 못해 좌절했습니다... 그래서 관련 논문을 본격적으로 살펴보기에 앞서 Object detection의 정의와 평가 방법 등에 대해 공부한 내용을 정리해보았습니다. 

## Object detection

![object detection](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2Fbv7d1Q%2FbtqufcdcRRK%2F6D21FiAhjEKGKSqbTx1aE1%2Fimg.png)
<p align="center">[그림 1] Object detection</p>

딥러닝을 활용한 컴퓨터 비전 영역의 문제는 객체의 수와 위치 판별 범위에 따라 달라집니다.  Image Classification은 이미지 내 단일 객체에 대해 label을 분류하는 작업입니다. 반면  **Object Detection**은 2개 이상의 객체에 대한 위치를 찾고(localize), 객체에 대한 label을 분류하는 작업입니다. 즉 Object detection은 다음과 같이 표현할 수 있습니다. 
<p align="center">**Object detection = Localization + Classification**</p>

객체의 위치는 객체가 포함된 가장 작은 크기의 사각형인 **bounding box**를 통해 파악할 수 있습니다. Object detection의 과제는 bounding box를 이루는 x,y 좌표값을 찾아내고 bounding box 내에 있는 객체를 분류하는 것이라고 할 수 있습니다. 추가적으로 객체를 탐지하는 것을 넘어 pixel 단위로 객체를 분류하는 Instance Segmentation 영역도 있지만 이에 대해서는 추후 포스팅에서 다루도록 하겠습니다. 

## Object detection Metric

다음으로 object detection이 잘 이뤄졌는지 수치적으로 파악할 수 있는 평가 지표에 대해 살펴보도록 하겠습니다. 현재 많은 Object detection 논문에서는 **mAP(means Average Precision)**를 주요 평가 지표로 활용하고 있습니다. 먼저 mAP에 대해 파악하기 전에 알아둬야 할 개념들부터 살펴보도록 하겠습니다. 

### 1. IoU(Intersection over Union)

<p align="center">![iou equation](https://camo.githubusercontent.com/1f5a5a29fd1d77bdd6c50f1dc422263d1a304b57/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354374657874253742494f552537442533442535436672616325374225354374657874253742617265612537442535436c656674253238425f25374270253744253230253543636170253230425f2537426774253744253230253543726967687425323925374425374225354374657874253742617265612537442535436c656674253238425f25374270253744253230253543637570253230425f25374267742537442532302535437269676874253239253744)</p>
<p align="center">![iou image](https://github.com/rafaelpadilla/Object-Detection-Metrics/raw/master/aux_images/iou.png)</p>
<p align="center">[그림 2] IoU의 수식과 도식</p>

**IoU(Intersection over Union)**는 객체의 위치 추정의 정확도를 평가하는 지표입니다. 실제 bounding box를 B_gt(ground truth)라고 하고, 예측한 bounding box를 B_p(predicted)라고 할 때 두 box가 겹치는 영역의 크기를 통해 평가하는 방식입니다. IoU는 0과 1 사이의 값을 가지며 Precision과 Recall 값을 결정하는데 임계값으로 사용됩니다. 


### 2. Precision and Recall


![precision and recall](https://ifh.cc/g/S4bhPr.jpg)
<p align="center">[그림 3] How to calculate precision and recall</p>

위의 그림을 보면 노란색 사각형은 실제(ground truth) bounding box이며, 빨간색 사각형은 object detection 알고리즘을 통해 예측한(predicted) bounding box입니다. 
- 1번 box는 ground truth bounding box와 거의 비슷한 위치와 크기를 보입니다. 이는 Object의 위치를 정확히 detect한 것이며 이같은 경우를 **TP(True Positive)**에 해당한다고 합니다. 

- 2번 box는 객체가 없음에도 detect한 경우로 **FP(False Positive)** 경우에 해당합니다.

- 3번 box는 ground truth bounding box와 어느 정도 겹치는 모습을 보입니다. 하지만 온전히 객체를 detect하지는 못했습니다. 객체의 detection 여부를 수치적으로 명확하게 하기 위해 앞서 살펴본 **IoU를 임계값(threshold)으로 사용**합니다. 예측한 bounding box의 IoU가 미리 지정한 임계값보다 높은 경우 TP에 해당하며 그렇지 않은 경우 FP에 해당한다고 볼 수 있습니다. 

- 위의 이미지에는 없지만 detect 되어야 했지만 그렇지 못한 경우는 **FN(False Negative)**에 해당합니다. 

지금까지 살펴본 TP, FP, FN를 조합하여 Precision과 Recall이라는 분류 성능 지표를 도출할 수 있습니다. 
<p align="center">![precision](https://camo.githubusercontent.com/7ce36f4e4fb2be6567a0c9ae9ab4f1d1b1e71288/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354374657874253742507265636973696f6e253744253230253344253230253543667261632537422535437465787425374254502537442537442537422535437465787425374254502537442b2535437465787425374246502537442537442533442535436672616325374225354374657874253742545025374425374425374225354374657874253742616c6c253230646574656374696f6e73253744253744)</p>
<p align='center'>![recall](https://camo.githubusercontent.com/ab16d220dae5b58815a6db9f0305c9e6b1c733fc/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354374657874253742526563616c6c253744253230253344253230253543667261632537422535437465787425374254502537442537442537422535437465787425374254502537442b25354374657874253742464e2537442537442533442535436672616325374225354374657874253742545025374425374425374225354374657874253742616c6c25323067726f756e64253230747275746873253744253744)</p>
<p align='center'>[그림 4] Precision과 Recall</p>

**Precision(정밀도)**은 전체 detect한 경우의 수 중 실제로 올바르게 detect한 비율을 의미합니다. **Recall(재현율)**은 이미지 내 객체의 수와 올바르게 detect한 비율입니다. 

![trade-off](https://ifh.cc/g/CjlKcH.jpg)
<p align='center'>[그림 5] Precision-Recall trade-off</p>

Precision이 높아지면 Recall이 낮아지고, Recall이 높아지면 Precision이 낮아집니다. 이 같은 관계를 **Precision-Recall trade-off**라고 합니다. 위의 그림은 스팸 메일(빨간색 메일)과 정상 메일(초록색 메일)을 분류하는 알고리즘의 성능을 Precision과 Recall을 통해 평가하는 도식입니다. Recall과 Precision이 서로 반비례 관계를 가지는 것을 확인할 수 있습니다. Precision과 Recall을 균형 있게 활용하여 Object detection 알고리즘의 성능을 평가하기 위해 AP라는 새로운 평가 지표가 등장합니다.

### 3. AP(Average Precision)

머신러닝 분야에서 Precision과 Recall 값을 그래프로 나타내 AUC(Area Under Curve)를 구해 분류 성능을 측정하는 것이 일반적입니다. 하지만 Object detection의 경우 하나의 이미지에 여러 객체가 detect될 수가 있으며 이는 객체마다 Precision Recall 그래프가 생성되는 것을 의미합니다. 시각화의 편이와 보다 종합적인 결과를 얻고자 Object detection 시 **AP(Average Precision)**이라는 평가 지표를 사용합니다. AP에 대한 설명은 다음 [github repo](https://github.com/rafaelpadilla/Object-Detection-Metrics)를 많은 부분 참고했습니다. 

![images](https://github.com/rafaelpadilla/Object-Detection-Metrics/raw/master/aux_images/samples_1_v2.png)
<p align='center'>[그림 6] Object detection 예시</p>

위의 그림은 각 이미지별로 객체를 detect한 결과를 보여주고 있습니다. 초록색 box는 실제 객체의 위치를, 빨간색 box는 Object detection 알고리즘에 의해 예상된 객체의 위치입니다. box 아래의 수치는 **confidence score**로 Object detection 알고리즘이 분류한 label에 대한 확신 정도를 의미합니다. 예를 들어 알고리즘이 (softmax 함수 등을 통해)특정 객체가 비행기일 확률이 0.88이라고 했을 경우 confidence socre는 88%라고 할 수 있습니다. AP는 아래의 과정을 통해 구할 수 있습니다. 

![process](https://ifh.cc/g/e9LX37.jpg)
<p align='center'>[그림 7] Precision-Recall graph를 구하는 과정</p>

(a) AP를 구하기 위해서 먼저 위의 도표와 같이 예측 bounding box별로 confidence score, TP, FP 여부로 분류합니다. Detections 컬럼은 image와 상관없이 bounding box를 개별적으로 파악하기 위해 알파벳을 임의로 붙였습니다.

(b) 과정 (a)에서 구한 도표를 confidence score에 따라 내림차순을 정렬합니다. 그리고 이전 열에 존재했던 TP와 FP의 수를 구해 누적(Accumulated) TP, FP 값을 구합니다. 이를 통해 bounding box별로 Precision과 Recall도 구할 수 있습니다. 

(c) 과정 (b)에서 구한 bounding box별 Precision, Recall 값을 그래프에 나타냅니다. 이 그래프를 통해 Recall값에 따라 변화하는 Precision 값을 확인할 수 있습니다.

![final graph](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision_v2.png?raw=true)
<p align='center'>[그림 8] PR곡선을 통해 AP 값을 구함</p>

마지막으로 계산의 편의를 위해 PR 곡선을 위와 같이 계단처럼 변형시켜줍니다. PR 곡선 아래의 면적이 AP값이며 개별 객체에 대한 Object detection 알고리즘의 성능을 의미합니다. 


### 4. mAP(mean Average Precision)

![map](https://ifh.cc/g/ULMZ3t.jpg)
<p align='center'>[그림 9] RCNN 논문에서 발췌한 mAP, AP metric 도표 </p>

AP가 여러 PR 곡선을 결과를 종합한 것과 유사하게 mAP(mean Average Precision)는 개별 객체에 대한 AP 값을 종합한 값입니다. 이름에서 알 수 있듯이 mAP는 Object detection시 검출한 객체별 AP 값의 평균입니다. 위의 도표는 bike, bird 등등 개별 객체에 대한 AP 값과 그 평균인 mAP 값을 보여주고 있습니다. 

### Conclusion
지금까지 Object detection의 과제와 평가 방법에 대해 살펴보았습니다. 지금까지 공부했던 Image classification과 많은 부분이 달라 Object detection에 대해 이해하는데 많은 시간이 걸렸습니다. 특히 metric 부분을 이해하는데 많은 사전지식이 필요했던 것 같습니다. 다음 포스팅에부터는 대표적인 Object detection 알고리즘에 대해 살펴보도록 하겠습니다!

### Reference
[Object detection metric에 대해 정말 잘 정리해놓은 github repo 강추!!!!!!](https://github.com/rafaelpadilla/Object-Detection-Metrics)  
[Object detection 전반에 대해 잘 설명한 유튜브 재생목록, 역시 강추!!!](https://www.youtube.com/watch?v=9I6nzfx_kpE&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=2&t=0s)   
[Object detection metric에 대해 깔끔하게 설명한 블로그](https://bskyvision.com/465)  






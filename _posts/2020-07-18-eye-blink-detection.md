---
layout: post
title:  "[ML]  Real-Time Eye Blink Detection using Facial Landmarks 논문 리뷰"
subtitle:   "eye blink detection"
categories: study
tags: ml
comments: true
use_math : true
---

최근 얼굴의 생동감(liveness)를 감지하는 프로젝트를 진행하고 있습니다. 앞서 살펴본 [texture and frequency analysis](https://herbwood.github.io/study/2020/05/31/texture-frequency-analysis/) 방법 외에도 인간의 미묘한 생체 활동 역시 생동감을 감지하기에 좋은 단서입니다. 그 중 눈 깜빡임은 모든 인간이 공통적으로, 무의식적으로 행하는 생체 활동입니다. 눈 깜빡임 여부를 알 수 있다면 생동감 역시 판단할 수 있다고 생각했습니다. 그래서 이번 포스팅에서는 눈 깜빡임 감지 알고리즘을 제시한 [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) 논문을 살펴보도록 하겠습니다. 

## What's the Problem?

기존에 눈 깜빡임을 감지하기 위해  눈의 위치를 찾아 눈이 눈꺼풀에 덮혀있는지 여부를 판별하는 방식을 사용했습니다.  하지만 이러한 방식은 주변 환경, 즉 조도, 각도, 해상도, 포즈 등에 따라 감지 성능이 크게 달라진다는 문제가 있습니다. 

<p align="center"><img src="http://t2m.kr/HLYjo" width="600px"></p><p align="center">[그림 1] Face Landmarks</p>

**facial landmark detector**는 사람의 안면 이미지에서 대부분의 landmark(눈, 코, 입 등 주요 안면 부위들)을 추출할 수 있습니다. 최신 facial landmark detector는 5% 이하의 오류율을 가지며 대부분 실시간 동작 가능한 성능을 보입니다. 본 논문에서는 face landmark를 활용하여 **SVM**을 통해 눈 깜빡임을 감지하는 방법을 소개합니다. 

## Improvements

### Description of features
<p align="center"><img src="http://media5.datahacker.rs/2020/05/39-1.jpg" width="400"/></p><p align="center">[그림 2] Eye Landmarks</p>

Face Landmark detector를 사용하면 안구의 주요 위치에 대한 랜드마크 역시 구할 수 있습니다. 위의 사진에서 볼 수 있듯이 68개의 랜드마크를 검출한다고 했을 때 안구는 총 안구 상단 2개, 하단 2개, 양옆끝 각각 1개씩 **6개의 랜드마크**가 검출됩니다. 이를 활용하여 눈의 깜빡임 여부를 감지하는 것이 가능합니다. 

논문에서는 얼굴 랜드마크 검출을 통해 눈 깜빡임 상태인 **눈의 종횡비율(Eye Aspect Ratio, EAR)** 을 도출합니다. EAR은 비디오의 각 프레임별로 구해집니다. EAR을 도출하는 공식은 아래와 같습니다. 

$$EAR = {||p_2 - p_6|| + ||p_3 - p_5|| \over 2||p_1-p_4||} $$

위의 공식을 통해 눈의 윗꺼풀과 아랫꺼풀의 절댓값을 기준으로 EAR이 결정됨을 알 수 있습니다. 윗꺼풀과 아랫꺼풀 사이의 간격이 작아지면, 즉 눈이 감기면 EAR 역시 0에 가까워지게 됩니다. 대신 눈을 뜬 상태인 경우 EAR은 대체로 같은 값을 유지합니다. 또한 눈 깜빡임은 양 눈 모두에서 동시에 행해지기 때문에 EAR 양 눈의 수치를 평균을 통해 도출합니다. 

### Classification

EAR값이 상대적으로 긴 시간동안 낮은 값을 유지하는 것은 눈 깜빡임을 의미하지 않습니다. 왜냐하면 눈을 의도적으로 오래동안 감거나, 하품을 하는 경우 눈을 감게 되기 때문입니다. 따라서 논문의 저자는 단일 프레임이 아닌 어느 정도 길이를 가진 프레임을 입력값으로 받아 눈 깜빡임을 감지하는 방식을 제시합니다. 실험을 통해 30fps(frame per second) 비디오의 경우 6프레임의 이미지를 입력으로 받으면 눈 깜빡임을 감지하기 쉽다고 합니다. 

이러한 눈 깜빡임을 감지하는 분류는 linear SVM(논문의 저자는 **EAR SVM**이라고 명명했습니다)을 통해 실행됩니다. 특정 프레임동안의 EAR 수치와 눈 깜빡임 여부(label)를 데이터셋으로 두고 EAR SVM을 통해 이진 분류를 하게 됩니다. 눈 깜빡임 데이터셋(positive dataset)과 눈을 깜빡이지 않는 데이터셋(negative dataset)을 통해 SVM을 학습시킵니다. 

<br>

## Experiments

### Accuracy of landmark detectors

앞서 살펴보았듯이 눈 깜빡임을 감지하기 위해 안구의 랜드마크를 사용하다보니 face landmark detector 자체의 성능이 상당히 중요합니다. 논문의 저자는 **Chehra**와 **Intraface**라는 최신 landmark detector을 비교하여 landmark detector의 정확도를 측정합니다. 정확도를 위해 사용한 loss 함수는 다음과 같습니다.

$$\eta={100 \over kN} \sum_{k=i}^N ||x_i - \hat{x_{i}}||$$

위의 수식에서 $x_i$는 랜드마크의 실제 위치(ground-truth location)이며, $\hat{x_{i}}$은 detector에 의해 예측된 랜드마크입니다. loss function은 두 랜드마크 사이의 Euclidean distance를 구한 것이라고 볼 수 있습니다. 논문의 저자는 위의 loss function을 **IOD(Inter-Ocular Distance)** 라고 부릅니다. 

<p align="center"><img src="https://ifh.cc/g/MeGBEM.png" width="400px"></p><p align="center">[그림 3] Chehra vs Intraface 성능 비교</p>

얼굴 전체의 랜드마크를 검출하는 경우, Chehra가 더 적은 에러를 보였으나, 안구 부분의 12개의 랜드마크만을 검출하는 경우 Intraface가 항상 더 좋은 성능을 보였습니다. 논문의 저자는 Intraface가 Chehra보다 작은 이미지에 보다 강건한 모습을 보인다고 합니다. 


### Eye blink detector evaluation

<p align="center"><img src="https://ifh.cc/g/CYvkdP.jpg" width="300px"></p><p align="center">[그림 4] Eye blink detector 성능 비교</p>

논문에서 제시안 방식은 기존 eye blink detector보다 월등한 성능을 보인다고 합니다. 그리고 EAR 수치에 threshold를 두고 눈 깜빡임 여부를 감지하는 방식(예를 들어 threshold=0.2라고 했을 때 0.2 이하일 때 눈 감은 상태로 판별하는 방식)과 비교했을 때 여전히 **EAR SVM**이 더 좋은 성능을 보였다고 합니다. 위의 그림에서 가장 위의 그래프는 시간에 따른 EAR 수치의 변화를 나타냅니다. EAR 수치가 낮아졌을 때 blink로 ground truth가 나타나는 모습을 확인할 수 있습니다. 그리고 EAR SVM이 ground truth와 상당히 유사한 결과를 보입니다. 

<br>

## Conclusion

본문에서는 언급하지 않았지만 Face Landmark 검출을 위해서 Haar-cascade 방식이 사용됩니다. 전통적인 컴퓨터 비전 분야에서 활용되는 방식으로 여전히 상당히 좋은 성능을 보여줍니다. 현재는 dlib, opencv에 내장pre-trained된 모델을 통해 빠르게 랜드마크 검출이 가능합니다. 신경망을 사용하지 않아도 이 정도로 좋은 성능을 보인다는 점이 흥미로웠습니다. 

## References
[Real-Time Eye Blink Detection using Facial Landmarks 논문](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)  
[eye blink detector 예제 코드](https://github.com/herbwood/study_datascience/blob/master/vision/detect_blinks.py)  
[Face Landmarks 그림 1 출처](http://t2m.kr/HLYjo)  
[eye blink detector에 대해 잘 설명해준 블로그](http://datahacker.rs/011-how-to-detect-eye-blinking-in-videos-using-dlib-and-opencv-in-python/)  



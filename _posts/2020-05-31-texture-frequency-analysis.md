---
layout: post
title:  "[ML] Face Liveness Detection based on Texture and Frequency Analysis 논문 리뷰 "
subtitle:   "texture frequency analysis"
categories: study
tags: ml
comments: true
use_math : true
---

최근 들어 **FIDO(Fast IDentity Online)** 방식을 통해 기존 ID, 패스워드가 아닌 홍채, 얼굴, 지문과 같은 생체 정보를 통해 인증하는 방식이 유행하고 있습니다. 특히 얼굴 인식 방식이 편리하다는 이유로 각광받고 있습니다. 모든 보안 문제가 그러하듯, 얼굴 인증 방식을 통과하려는 악의적인 침입인 **Spoofing** 역시 등장하였습니다. 얼굴 인증을 통과하는 Spoofing은 가상의 얼굴을 대조함으로써 인증을 통과하는데 구체적인 방법은 아래와 같습니다. 

<p align="center"><img src="https://miro.medium.com/max/671/0*FhPaS2RPMYUYULjC" width="500px"></p>
<p align="center">[그림 1] Face Spoofing 방법의 종류</p>

얼굴 인증 Spoofing은 얼굴 사진, 실시간 얼굴 영상, 3D 마스크를 통한 방법이 있습니다. 실제로 기존 얼굴 인식 인증 방식은 Face Spoofing을 통해 인증을 허가하는 문제가 있습니다. 하지만 이러한 Spoofing 인증 방식은 공통적으로 실제 얼굴에서 나타나는 **생동감(Liveness)** 가 없습니다. 이번 포스팅에서는 Face Spoofing 방지를 위해 texture(질감)와 frequency(주파수) 분석을 통해 Face Liveness를 감지하는 논문에 대해 살펴보도록 하겠습니다. 

<br>

## Difference between images taken from live faces and 2-D paper mask

논문에서는 실제 얼굴과 안면 마스크의 차이에서 얼굴 생동감 분류에 대한 해답을 찾습니다. 먼저 실제 얼굴과 고정된 얼굴 이미지는 대표적으로 두 가지 차이가 있습니다.  

1) 실제 얼굴이 고정된 사진 이미지보다 **입체감(3-D shape)** 면에서 다릅니다.
2) 또한 고정된 얼굴 이미지는 촬영 후 다시 출력한 결과이기 때문에 **세밀함(detailness)** 면에서 실제 얼굴보다 뒤떨어집니다. 

입체감 여부는 **저주파 영역(low frequency region)** 의 차이를 불러오고 이는 전체 얼굴 형태의 **조도 요소(illuminance component)** 와 큰 연관이 있습니다. 세심함의 차이는 고주파 정보의 차이를 불러일으킵니다. 뿐만 아니라 고정된 얼굴 이미지의 경우 실제 얼굴에 비해 질감의 풍족함이 떨어집니다. 

<br>

## Proposed method

### Frequency-based feature extraction

논문의 저자는 먼저 주파수 기반으로 이미지의 특징을 추출하는 방법에 대해 소개합니다. 방법은 아래와 같은 순서로 진행됩니다. 

<p align="center"><img src="https://ifh.cc/g/wsLqN3.png" width='400px'></p>
<p align="center">[그림 2] Frequency-based feature extraction</p>

1) 얼굴 이미지는 로그가 적용된 푸리에 변환(**log scaled Fourier transformation**)을 통해 주파수 도메인(frequency domain)으로 변환합니다(이미지에서 주파수란 이미지가 변화하는 정도를 의미합니다).
 
2) 푸리에 변환된 이미지는 주파수가 0인 요소가 중심으로 가도록 이동시킵니다. 

3) 변환된 결과는 동심원 고리(**concentric ring**)의 형태로 나눕니다. 각 고리의 반지름이 1이 되도록 만듭니다. 만약 변환된 이미지의 크기가 64 x 64라면 32개의 동심원 고리가 생기게 됩니다(반지름이 작다는 것은 이미지에 대한 저주파 정보를 담게되는 것을 의미합니다).

4) 마지막으로 모든 동심원 고리가 가지는 값들에 대한 평균값 연결시키고 min-max normalization을 적용하여 정규화를 진행한 후 **1-D feature vector**를 얻습니다. 

### Texture-based feature extraction

<p align="center"><img src="https://ifh.cc/g/QAJC6n.png" width='400px'></p>
<p align="center">[그림 3] Texture-based feature extraction</p>

다음으로 질감 기반의 특징 추출(texture-based feature extraction) 방법에 대해 살펴보겠습니다. 논문의 저자는 질감 분석을 위해 **LBP(Local Binary Pattern)** 기법을 사용합니다. 기법의 순서는 아래와 같습니다. 

<p align="center"><img src="https://t1.daumcdn.net/cfile/tistory/99817F465A655D9E22"></p>
<p align="center">[그림 4] Local Binary Pattern</p>

1) 3 x 3 크기의 grid에서 중심에 위치한 픽셀의 값과 나머지 8개의 픽셀의 값을 비교합니다. 
2) 중심 픽섹값보다 작은 경우 0, 크거나 같은 경우 1로 **encoding** 해줍니다. 
3) encoding된 결과를 반시계방향으로 읽어들여 **이진수 값(Binary value)** 을 얻습니다
4) 얻은 이진수 값을 **십진수 값(Decimal value)** 로 변환해줍니다.
5) 이를 모든 픽셀에 대해 적용한 후 각 십진수값에 대한 빈도수를 얻어 **1-D feature vector**를 얻습니다. 

<p align="center"><img src="https://ifh.cc/g/QIKbWt.png" width="400px"></p>
<p align="center">[그림 5] P, R 값 변화에 따른 시각화</p>

주변 픽셀을 어디까지 encoding할지 여부에 따라 bin 값이 달라질 수 있습니다. 위에서 살펴본 예시와 같이 주변 8개의 픽셀에 대해서만 encoding할 경우 00000000(이웃한 픽셀값이 중심 픽셀값보다 모두 작은 경우)~11111111(이웃한 픽셀값이 중심 픽셀값보다 크거나 같은 경우)까지 총 256 크기의 feature vector를 얻을 수 있습니다. 논문에서는 실험 결과 이웃한 픽셀의 수 P=8, 중심에 위치한 픽셀과 이웃한 픽셀과의 거리인 R=1로 설정하였습니다.  


### Fusion-based feature extraction

논문에서는 frequency-based feature extraction 방식과 texture-based feature extraction 방식을 혼합하여 최종 1-D feature vector를 추출합니다. 추출된 결과를 SVM(Support Vector Machine)을 통해 학습시켜 생동감을 분류하는데 사용합니다.  

<br>

## Experiments

<p align="center"><img src="https://ifh.cc/g/dQlUd7.jpg"></p>
<p align="center">[그림 6] 사용한 데이터셋</p>

- BERC Webcam Database와 BERC ATM Database를 데이터셋으로 사용하였습니다. 해당 데이터셋은 실제 얼굴과 사진, 출력된 사진, 잡지, 캐리커쳐와 같은 가짜 얼굴 이미지로 구성되어 있습니다. 

- Webcam database에서 683장의 실제 안면 이미지, 3628의 가짜 얼굴 이미지를 사용하였으며, ATM database에서 897장의 실제 안면 이미지와, 2798장의 가짜 얼굴 이미지를 사용했습니다. 

<p align="center"><img src="https://ifh.cc/g/RYWhRM.png" width="400px"></p>
<p align="center">[그림 7] 실험 결과</p>

- SVM 분류기의 경우 RBF kernel이 사용되었으며 기타 파라미터는 genetic algorithm에 의해 최적화 되었습니다. 

- frequency-based 방식과 texture-based 방식은 비슷한 성능(각각 **EER(Equal Error Rate)** of 11.87%, 11.58%)을 보였으나 두 방식을 혼합한 방식(fusion-based)은 훨씬 더 좋은 성능(**EER of 8.48%**)을 보였습니다. 

- 서로 다른 두 방식은 데이터셋에 따라 성능 차이가 존재했습니다. 

<br>

## Conclusion

지금까지 frequency와 texture 분석을 통해 얼굴의 생동감을 판별하는 방법에 대해 살펴보았습니다. 논문 자체에 전통적인 컴퓨터 비전 분야에서 자주 사용되는 기법이나 이론적인 내용들이 있어 읽는데 조금 어려움이 있었습니다. 하지만 얼굴 인식 분야에서는 여전히 Haar-cascade와 같이 딥러닝 기반이 아닌 전톤적인 컴퓨터 비전 기법들이 사용되고 있어, 이번 기회에 공부할 수 있었던 좋은 기회가 되었던 것 같습니다. 현재 얼굴 생동감을 판별하는 프로젝트를 진행할 예정인만큼 향후 포스팅에서도 관련된 논문에 대한 리뷰를 올려보도록 하겠습니다.

<br>

## Reference
[Face Liveness Detection based on Texture and Frequency Analysis 논문](https://web.yonsei.ac.kr/jksuhr/papers/Face%20Liveness%20Detection%20Based%20on%20Texture%20and%20Frequency%20Analyses.pdf)  
[이미지와 주파수 사이의 관계를 잘 설명해주신 백곳님 블로그](https://idlecomputer.tistory.com/120)    
[푸리에 변환에 대해 잘 설명해주신 다크 프로그래머님 블로그](https://darkpgmr.tistory.com/171)  
[LBP에 대해 잘 설명해주신 스카이비전님의 블로그](https://bskyvision.com/280)  

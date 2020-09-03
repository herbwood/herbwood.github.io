---
layout: post
title:  "[Project] 작물 병충해 이미지 분류 서비스 CropsDoctor"
subtitle:   "cropsdoctor"
categories: study
tags: project
comments: true
use_math : true
---

이번 포스팅에서는 제가 현재 참여하고 있는 교육기관에서 진행한 프로젝트에 대해 소개하도록 하겠습니다. 교육기관에서 2달 동안 컴퓨터 공학 및 머신러닝 강의를 수강 후 일 주일 동안 팀원과 짧게 미니 프로젝트를 진행하게 되었습니다. 

## 프로젝트 개요

프로젝트 주제는 UN이 제시하는 **지속 가능한 발전 목표(SDGs)** 중에 선정해야 했고, 제가 속한 팀은 "기아 해결" 문제를 해결하기 위해 작물 병충해 이미지 분류 서비스를 기획하게 되었습니다. 병충해가 발생하면 작물의 수확량이 크게 떨어지고 이에 따라 식량난이 발생할 수 있습니다. 이러한 사태를 미연에 방지하기 위해 작물의 병충해 감염 여부를 파악하는 것이 중요하다고 생각하여, 저희 조는 작물 잎 이미지 분석을 통해 병충해 감염 여부와 적절한 정보를 제공하는 서비스를 프로젝트로 진행하게 되었습니다. 접근성을 위해 단순히 PC에서만 구동되는 것이 아닌, 실제 안드로이드 기기에서 작동하는 스마트폰 어플리케이션 개발을 최종 목표로 정했습니다. 

<br>

## 프로젝트 진행 과정

### 데이터 수집

병충해 감염 여부는 작물을 잎을 통해 쉽게 확인할 수 있습니다. 저희 팀에서는 수집에 앞서 어떤 작물의, 어떤 병충해와 관련된 데이터를 수집할지부터 정했습니다. 먼저 농업과 병충해에 관련된 정보를 제공해주는 [농사로](http://www.nongsaro.go.kr/portal/ps/psx/psxk/fixFarmTchSch.ps?menuId=PS04315&totalSearchYn=Y&dbyhsNm=%EC%A0%90%EB%B0%95%EC%9D%B4%EC%9D%91%EC%95%A0&agchmNm=&kidoNm=%ED%86%A0%EB%A7%88%ED%86%A0#none) 사이트를 참고하여 병충해에 자주 노출되는 작물과 해당 작물이 자주 노출되는 병충해를 선정했습니다. 

<p align="center"><img src="https://camo.githubusercontent.com/0a227ad5f1914ec5484a7aa683bb7c5a34c35012/68747470733a2f2f6966682e63632f672f304b66674c312e706e67"></p><p align="center">[그림 1] 수집한 작물별 병충해 이미지 예시</p>


- 저희는 토마토, 감자, 옥수수, 콩, 고추를 대상 작물로, 세균점무늬병, 겹둥근무늬병, 균핵병 등 29종의 병충해를 선정했습니다. 병충해 감염 여부를 판별하기 위해서는 작물별 건강한 잎 이미지도 필요하기 때문에 총 34개의 label(=29개의 병충해에 감염된 잎 이미지 + 작물 5종에 대한 건강한 잎 이미지)을 가지는 데이터셋을 구축하기로 결정했습니다. 

- 저희는 먼저 Kaggle에서 공개한 [plant disease dataset](https://www.kaggle.com/saroz014/plant-diseases)을 통해 필요한 대부분의 데이터를 수집했습니다. 

- 부족한 데이터에 대해서는 [Google image crawler](https://ecsimsw.tistory.com/entry/Crawling-Scraping?category=869268)를 활용하여 데이터를 추가적으로 수집했습니다. 

### 데이터 전처리

- 256 x 256 크기로 resize해줬습니다
- class별로 데이터의 수가 불균형하여 data augmentation을 적용하여 비슷한 수준으로 맞춰줬습니다. 

### 모델 학습

- 작물 잎 이미지를 통해 병충해 감염 여부 및 병명 분류는 이미지 분류 문제로 정의하였습니다. 다양한 CNN 모델을 사용해보고, 하이퍼파라미터를 조정하여 성능을 실험하는 과정을 거쳤습니다. 
- 전체 데이터셋에서 90%를 훈련셋으로 사용했습니다.
- tensorflow 프레임워크를 통해 학습시켰습니다.
- Google Colab 및 Kaggle Notebook의 GPU를 활용하여 실험했습니다. 
- 실험 결과 **최종 91.69%의 정확도와 loss 1.0403**의 성능을 보이는 모델을 학습시켰습니다. 

#### 모델 탐색

<p align="center"><img src="https://ifh.cc/g/mm82nC.png" width="600px"></p><p align="center">[그림 2] 모델에 따른 epoch별 정확도 및 loss </p>

- 모델은 ResNet-50, MobileNet v2, EfficientNet lite0~4까지 총 7종의 모델을 통해 분류 성능을 측정했습니다.
- epoch(=20) 및 기타 하이퍼 파라미터를 고정하고 다양한 모델의 정확도와 loss를 실험했습니다.
- 실험 결과 **Efficientnet lite0**으로 학습 시 정확도는 89.68%, loss는 1.0911로 가장 좋은 성능을 보였습니다.
- 반면 ResNet-50은 정확도 86.60%, loss 1.1665로 가장 낮은 성능을 보였습니다. 

#### 하이퍼파라미터 탐색

<p align="center"><img src="https://ifh.cc/g/DmmqJV.png" width="600px"></p><p align="center">[그림 3] epoch별 정확도 및 loss </p>

- 모델을 EfficientNet lite0으로 사용하고 기타 하이퍼 파라미터를 고정하고 epoch을 50까지 늘려본 결과 40정도에서 epoch이 수렴하는 결과를 보였습니다.
- **epoch=40**으로 지정 시 정확도 91.31%, loss는 1.0658이라는 결과를 보였습니다. epoch=50 시 정확도가 및 loss가 수렴하였습니다. 

<p align="center"><img src="https://ifh.cc/g/xBkWiy.png" width="600px"></p><p align="center">[그림 4] dropout rate 및 use augmentation 실험 결과</p>

- 이외에도 dropout의 비율을 다르게 하여 validation accuracy를 실험한 결과 dropout rate=0.2일 때 정확도 90.68%로 가장 좋은 결과를 보였습니다. 
- 모델에 내장된 추가적인 data augmentation을 실행해주는 use_augmentation 파라미터의 경우 추가적인 augmentation을 적용한 경우 오히려 성능이 하락하는 보습을 보였습니다. 제 개인적인 생각으로, 작물의 잎이라는 데이터의 특성상 다양한 augmentation, 가령 색상 반전 등을 적용하면 모델의 학습을 방해하기 때문인 것 같습니다. 

### 모델 경량화

일반적으로 딥러닝 라이브러리는 용량을 많이 차지하며, 커다란 computing resources를 요구하기 때문에 일반적으로 스마트폰이나 IoT 기기에 내장될 수 없습니다. 이를 해결하기 위해 저희 팀은 학습된 모델을 경량화시켜주는 프레임워크인**Tensorflow Lite**를 사용하였습니다. Tensorflow Lite의 **Converter**를 통해 학습된 모델을 tflite 형식으로, label은 txt 형식으로 변환하였습니다. 이는 추후 모바일 플랫폼에 이식시켜 스마트폰 기기를 통해 촬영된 병충해를 분류하는데 사용됩니다. 

### 모바일 플랫폼 이식

<p align="center"><img src="https://ifh.cc/g/fird6H.jpg" width="600px"></p><p align="center">[그림 5] Android Studio 작업 화면</p>

- 저희 팀은 스마트폰을 통해 작물의 잎을 영상으로 촬용하면, 병충해 감염 여부와 병충해 감염 시 관련 정보 및 방제법을 제공하는 스마트폰 어플리케이션을 기획했습니다. 이를 구현하기 위해 Android Studio를 사용했습니다. 
- Tensorflow Lite로 경량화한 모델을 사용하기 위해 Tensorflow Lite 라이브러리르 설치했습니다.
- 경량화시킨 models.tflite, labels.txt 파일은 `app/src/main/assets` 디렉터리에 넣었습니다. 
- 모바일 기기를 통해 촬영되는 여상에 나오는 작물의 병충해 label을 labels.txt를 통해 읽어왔습니다.
- 그리고 각 label에 속한 확률을 화면 상에 표기했습니다.
- 꾸준히 Android Virtual Device를 통해 결과를 확인하면서 작업을 진행했습니다. 

<p align="center"><img src="https://ifh.cc/g/uEmBYx.png" width="300px"></p><p align="center">[그림 6] 앱 아이콘 및 로고</p>

- 저희 팀은 병충해 문제를 해결하여 작물들의 의사가 된다는 의미를 담아 앱 이름을 CropsDoctor라고 지었습니다.
- 앱 상단에 있는 로고 이미지와 저희의 프로젝트 기획 의도가 잘 담긴 앱 아이콘 역시 제작 시  포함시켰습니다. 


### 최종 결과

<p align="center"><img src="https://camo.githubusercontent.com/76955e36c55f6a232575486a4b0263bb52f89033/68747470733a2f2f6966682e63632f672f7961387358622e6a7067" width="600px"></p><p align="center">[그림 6] CropsDoctor 시연 화면</p>

- 최종 프로젝트 스마트폰 앱 **CropsDoctor**시연 화면은 위와 같습니다. 
- 메인화면에서 작물의 잎을 영상을 통해 촬영하면 감염되었을 확률이 가장 높은 병충해 명과 감염 확률을 보여줍니다(건강하다고 판단한 경우 healthy라고 표기됩니다). "방제 방법 알아보기" 버튼을 누르면 병충해에 관한 정보 및 방제 방법을 보여주는 화면으로 전환됩니다.
- 방제 방법은 앞서 언급한 농사로를 참고하였습니다. 

<br> 

## 결론

일주일만에 진행되었음에도 상당히 만족스러운 결과가 나왔던 것 같습니다. 

- 협업 툴을 많이 사용했습니다. Notion을 통해 스크럼 형식으로 프로젝트를 진행하여 효율적인 분업이 가능했습니다. 또한 github를 통해 각자 기여한 부분에 대한 contribution을 명확히 남겼고, 문서화도 상당히 잘 진행되었습니다. 

- 의외로 앱을 구현하는 부분에서 애먹었습니다. 저를 포함한 팀원 대부분이 앱 개발이 생소했고, 단순한 버튼을 구현하는데도 많은 시간이 소요되었습니다. 그래서 아무래도 UI에서 개선할 점이 있다고 생각하지만, 일주일이라는 짧은 시간 내에 보기 좋은 결과가 나왔던 것 같습니다. 

앱 개발이 미숙함에도 어플리케이션을 만들고 싶었던 이유는 폐쇄적인 환경에서 진행했던 딥러닝 실습으로부터 벗어나고 싶었기 때문입니다.  단순히 google colab에만 돌아가는 프로그램이 아닌 현실 세계와 맞닿아 있는 인공지능 어플리케이션을 꼭 만들고 싶었습니다. 부족하지만 구동되는 인공지능 어플리케이션을 만들어봄으로써 인공 지능을 현실 세계에 적용해볼 수 있었던 유의미한 프로젝트였던 것 같습니다. 

## 참고자료

[프로젝트 github](https://github.com/herbwood/crops_doctor)  
[병충해 정보 제공 사이트 농사로](http://www.nongsaro.go.kr/portal/ps/psx/psxk/fixFarmTchSch.ps?menuId=PS04315&totalSearchYn=Y&dbyhsNm=%EC%A0%90%EB%B0%95%EC%9D%B4%EC%9D%91%EC%95%A0&agchmNm=&kidoNm=%ED%86%A0%EB%A7%88%ED%86%A0#none)   
[Kaggle plant disease dataset](https://www.kaggle.com/saroz014/plant-diseases)  
[Google image crawler](https://ecsimsw.tistory.com/entry/Crawling-Scraping?category=869268)  
[Tensorflow Lite](https://www.tensorflow.org/lite/guide)  
[Tensorflow Lite Android Studio 내장하는 방법](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0)      

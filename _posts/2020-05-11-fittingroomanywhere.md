---
layout: post
title:  "[Project] 의류 가상 시착용 서비스 Fittingroomanywhere"
subtitle:   "fittingroomanywhere"
categories: study
tags: project
comments: true
use_math : true
---

이번 포스팅에서는 양재 혁신허브 R&CD에서 AI 교육과정에서 수행한 프로젝트를 소개하도록 하겠습니다. 해당 교육과정에서 2달간 머신러닝 강의를 수강하면서 팀 프로젝트를 진행하였습니다. 팀원들 모두 컴퓨터 비전 분야에 관심이 있어 관련 프로젝트 주제를 찾던 도중 수업 시간에 알게 되었던 GAN 모델을 활용하는 방향으로 주제를 잡게 되었습니다. 

## 프로젝트 개요

### 프로젝트 소개

<p align="center"><img src="https://ifh.cc/g/ZCzMb9.jpg"></p>
<p align="center">[그림 1] 프로젝트 개요 </p>

저희 조가 진행한 프로젝트는 쇼핑몰 의류 가상 시착용 서비스인 **Fittingroomanywhere**입니다. 의류 쇼핑몰에서 쇼핑을 할 경우 의류를 직접 입어보지 못해 자신과 어울리는지 제대로 파악하지 못한다는 단점이 있습니다. 저희는 실제 소비자가 자신이 쇼핑몰에서 착용하기를 원하는 의류를 고객이 가상으로 착용한 이미지를 제공하는 서비스를 목표로 프로젝트를 진행했습니다. 프로젝트명은 "어디서든지 피팅룸이 되어주겠다"라는 의미를 담아 지었습니다(ㅎ...).

### 프로젝트 구현 방안

이러한 서비스를 구현하기 위해 고객의 신체로부터 옷 부분만 분리한 후 착용을 희망하는 의류의 색상과 패턴을 입혀 다시 합성하는 방식을 고안했습니다. 이를 구현하기 위해 저희 조는 두 가지 딥러닝 모델을 활용하였습니다. 먼저 전체적인 프레임워크는 아래와 같습니다. 

<p align="center"><img src="https://ifh.cc/g/tacFfT.jpg"></p>
<p align="center">[그림 2] 프로젝트 프레임워크 </p>

- 먼저 Mask R-CNN 모델을 통해 고객의 신체에서 옷 부분만 image segmentation을 통해 mask를 얻어냅니다
- CycleGAN 모델을 통해 mask를 착용을 희망하는 의류의 패턴과 색상에 맞게 변환합니다. 
- 마지막으로 변환된 mask를 고객의 이미지에 맞게 image rendering해주는 과정을 거칩니다. 

세부적인 사항은 아래에서 살펴보도록 하겠습니다. 앞서 언급했다시피 프로젝트 설계 시 두 딥러닝 모델을 사용하기로 계획하였고, 효율적인 분업을 위해 R&R로 모델별로 작업 인원을 분배하였습니다. 먼제 각 모델별로 데이터 수집, 데이터 전처리, 모델 학습 과정 및 개선 방안을 본 후 세부 프레임워크를 살펴보도록 하겠습니다. 

<br>

## Mask R-CNN

저희는 image segmentation을 위해 **Mask R-CNN**이라는 모델을 사용하기로 했습니다.  상세한 작동 원리는 [해당 포스팅](https://mylifemystudy.tistory.com/82?category=797525)을 참고하시길 바랍니다. Mask R-CNN을 활용하여 고객의 이미지, 쇼핑몰 이미지에서 각각 의류 부분만 segmentation을 진행할 예정입니다. 

####  데이터 수집

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpjCpD%2FbtqxqJeofdT%2FdqQ0RGDVhGzMQFyfaGpf71%2Fimg.png"><</p>
<p align="center">[그림 3] Deep Fashion2 Dataset </p>

Mask R-CNN을 학습시키기 위해 의류에 대한 방대한 양의 데이터와 annotation을 포함하고 있는 [Deep Fashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)을 활용하였습니다. 이미지 파일 하나에 대한 Annotation 파일이 있으며, 이 annotation에는  item의 카테고리, segmentation 정보, landmark정보, keypoint정보 등이 포함되어 있습니다. 

 
#### 데이터 전처리

먼저 Mask R-CNN에서 사용하는 형식과 Deep Fashion2 에서 제공하는 형식이 달라 annotation 형식을 변환해주는 데이터 전처리 과정을 진행하였습니다. COCO 형식의 annotation 형식을 VIA 형식의 polygon 형식으로 변환하였습니다. 

#### 모델 학습


<p align="center"><img src="https://camo.githubusercontent.com/8e90658ec2534da2cb0afedce49c295e11f18066/68747470733a2f2f6966682e63632f672f6d5a6c6b43332e6a7067"></p>
<p align="center">[그림 4] Mask R-CNN 모델 학습 </p>

- Mask R-CNN 코드는 [해당 github 저장소](https://github.com/matterport/Mask_RCNN)를 clone하여 진행하였습니다.  
- 학습-훈련 데이터셋의 수를 각각 296장, 54장으로 했을 때 가장 좋은 성능을 보였습니다. 
- annoatation의 경우 landmark가 더 촘촘하게 갖춰진 segmentation 형식을 사용했을 때, 그리고 네트워크 하단에 layer를 3개 추가했을 때, epochs=30으로 지정했을 때 가장 좋은 결과를 보였습니다. 
- 이같은 과정을 통해 원본 이미지로부터 의류 부분만 분리하기 위한 Image segmentation 모델을 학습하였습니다. 

<br>

## CycleGAN

**CycleGan** 은 기존 Gan 에서 생성된 이미지를 원래 이미지로 되돌리는generator 가 추가된 모델입니다.  직관적으로 보자면 CycleGAN은 원래 이미지로 돌아갈 수 있는 선에서 이미지를 변화시키기 위해 사용되었다고 이해하시면 됩니다. 해당 모델을 사용하여 고객의 원본 의류의 크기를 유지한 채로 의류의 색상과 패턴을 변화시킬 수 있었습니다. 

<p align="center"><img src="https://ifh.cc/g/zeuuXr.jpg"></p>
<p align="center">[그림 5] 흑발을 금발로 바꾸는 CycleGAN 모델 구조 </p>

위의 그림은 본격적인 프로젝트 진행하기에 앞서 CycleGAN을 사용하여 흑발 연예인을 금발로 바꿔주는 프로그램을 작성한 결과입니다. 적은 양의 데이터로 학습시켜 최선의 결과가 나오지는 않았지만 머리 부분 위주로 색상이 변환된 것을 확인할 수 있습니다. 


#### 데이터 수집
저희는 의류 이미지 수집을 위해 [구글 이미지 크롤러](https://ecsimsw.tistory.com/entry/Google-image-crawler-Crawling-Scraping-python)를 사용하여 의류 데이터를 수집했습니다. Mask R-CNN 모델 학습 데이터와 달리 의류만 나온 이미지 찾고자 하여 데이터 수집하는 과정이 상당히 어려웠습니다. 데이터 수집 과정에서 캡쳐 도구;를 사용하여 이미지를 직접 수집하기도 했습니다. 


#### 데이터 전처리
CycleGAN에 학습하기에 앞서 이미지의 크기를 256 크기로 맞춰줬으며, 부족한 학습 데이터 수를 늘리기 위해 horizontal flip을 하여 data augmentation을 적용했습니다. 


#### 모델 학습

<p align="center"><img src="https://ifh.cc/g/lQ4RSY.png"><</p>
<p align="center">[그림 6] epoch에 따른 CycleGAN 모델 학습 결과 </p>

<p align="center"><img src="https://ifh.cc/g/eKR3fz.png"></p>
<p align="center">[그림 7] 학습 결과 </p>

- [그림 6]은 흰 색 티셔츠를 크기를 유지한 채로 파란색 티셔츠로 변환되는 과정을 보여줍니다. epoch=50으로 설정하였습니다. epoch이 낮을 때는 학습이 제대로 되지 않아 노이즈가 많으나 epoch이 늘어남에 따라 학습이 잘 진행되었습니다. [그림 7]은 epoch을 50까지 진행한 후의 최종 결과입니다. 의류의 크기와 주름은 유지되었고 색상만 바뀐 결과를 확인할 수 있습니다. 
- learning rate decay=0.009
- Identity loss를 조정하여 가중치를 20step 당 한번씩만 텀을 주고 학습시켜 스타일(색상, 패턴) 반영 정도를 높혔습니다. 
-   loss function으로 Smooth L1 loss를 사용했습니다. 


<br>

## 최종 프레임워크

<p align="center"><img src="https://ifh.cc/g/Bp3rtm.jpg" width="800px"></p>
<p align="center">[그림 8] 최종 프로젝트 프레임워크 </p>

1)  사용자가 착용하기를 희망하는 style image에 대해  segmentation을 적용합니다.   
2)  style image를 색상, 패턴에 따라 분류한 후 해당하는 분류에 따른 weight값을 load합니다.   
3)  사용자가 업로드한 이미지를 segmentation하여 segmentation한 이미지, mask, bounding box에 대한 정보를 얻습니다.  
4) segment된 유저 이미지를 CycleGAN 모델에 적용하기 위해 256 사이즈로 resize해줍니다.  
5)  resize된 이미지와 style image에 대한 weight값을 적용하여 새로운 이미지를 생성합니다.  
6)  새롭게 생성된 이미지를 원본 이미지로 resize해줍니다.  
7)  bounding box 사이즈만큼 crop해줍니다.  
8)  그리고 원본 이미지 사이즈만큼 padding을 추가합니다.  
9)  user image와 새롭게 생성된 이미지에 mask를 적용시켜줍니다.  
10)  두 이미지를 합쳐 최종 output을 반환합니다.   

<br>


<br>

## 결론

첫 팀 프로젝트였던만큼 어려웠지만 많은 것을 배울 수 있었습니다. 

- 깔끔하게 정제된 데이터를 사용하는 것이 아니었기에 직접 현실의 데이터를 수집하고 전처리 과정부터 상당히 번거로웠습니다. 데이터 수집 과정이 상당히 많은 시간이 소요되고 정제된 데이터를 찾는 과정의 중요성을 알게 되었습니다.  

- Google colab을 통해 cell별로 코드를 실행하는 것이 아니라 파일별로, 모듈별로 코드를 실행시키는 방식에 좀 더 익숙해지게 되었습니다. 딥러닝 프로젝트와 같이 작지 않은 규모의 코드 관리는 모듈별로 분리하여 관리하는 것이 더 효율적이라는 것을 알게 되었습니다. 

- 소프트웨어 개발 방법론 중 애자일(Agile)의 필요성을 알게 되었습니다. 빠르게 개발하고, 결과를 확인하고 개선해나가는 과정을 거치지 않고 폭포수 방식을 통해 프로젝트를 진행하여 프로젝트 후반부이 되어서야 이전 단계에서의 실수를 발견하게 되엇습니다. 

<br>

## 참고 자료

[프로젝트 github 저장소](https://github.com/herbwood/FittingroomAnywhere)    
[Deep Fashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)    
[참고한 Mask R-CNN 코드](https://github.com/matterport/Mask_RCNN)    
[팀원이 작성한 Mask R-CNN 관련 내용 블로그](https://mylifemystudy.tistory.com/82?category=797525)      
[구글 이미지 크롤러](https://ecsimsw.tistory.com/entry/Google-image-crawler-Crawling-Scraping-python)    

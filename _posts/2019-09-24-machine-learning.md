---
layout: post
title: "machine learning"
subtitle: "machine learning"
categories: study
tags: ML
comments: true
---

# 1. 머신러닝(Machine Learning)이란

## what? 
##### 머신러닝의 정의
input data에 대한 output 값을 명시적으로 프로그래밍하지 않아도 input data를 학습하여 가장 합리적인 output 값을 출력하도록 컴퓨터를 학습시키는 연구 분야.

## why? 
##### 머신러닝이 필요한 이유
통신 기술 및 SNS의 발전으로 생산되는 데이터의 수가 기하급수적으로 늘어났다.  
  
이에 따라     
1) 입력값에 대한 출력값을 일일히 지정해주는 기존의 방식의 한계가 나타났으며, 2) 다량의 데이터에서 유의미한 패턴을 찾아내는 것이 가능해졌다.     
  
이러한 배경 하에 기존 솔루션의 한계를 극복하고 데이터를 활용하여 시장성을 확보하기 위해 머신러닝의 필요성이 대두되었다. 

## how?
##### 1) by Model 
앞서 머신러닝은 기존 input data를 통해 컴퓨터를 학습시킨다고 언급하였다. 구체적으로 컴퓨터를 학습시킨다는 것은 기존 input data를 활용하여 데이터의 패턴을 파악하고 데이터의 패턴을 가장 잘 드러내는 가설인 ***Model***을 생성하는 것을 의미한다. 컴퓨터는 학습 과정을 거친 후 Model에 새로운 input data를 넣어 가장 합리적인 output을 출력하게 된다. 
<img src = 'C:\Users\default.DESKTOP-S5Q9GAA\Documents\Programs\herbwood.github.io\assets\img\model.png'></img>

##### 2) by optimization
그렇다면 어떤 방식으로 데이터의 패턴을 가장 잘 포착한 Model을 만들어낼 수 있을까?   
먼저   
1) input data 특성에 맞는 가상의 Model을 생성한 후   
2) 가상의 Model에 input data의 패턴을 잘 포착했는지 여부를 파악하는 optimization metrics를 적용한다.   
3) 파라미터를 수정 작업을 통해 optimization metrics에서 높은 점수를 받는 최적의 Model을 생성해낸다. 

##### 3) Machine Learning process
머신러닝을 적용하는 과정은 다음과 같다.
1. 문제 정의 및 평가 지표 선택
2. 데이터 수집
3. 탐색적 데이터 분석
4. 데이터 전처리
5. 모델 선택 및 적용
6. 성능 평가

##### 4) Diffrent kinds of Machine Learning Algorithms
- 머신러닝 알고리즘은 두 가지로 나눠질 수 있다. 머신러닝은 모델을 학습시킬 때 input data에 대한 label, 즉 답이 있는 경우 ***Supervised Learning(지도 학습)***이며, label이 없는 경우를 ***Unsupervised Learning(비지도 학습)***이라고 한다.   
- Supervised Learning은 다시 label의 종류에 따라 두 가지로 나뉘어진다. label이 continuous value인 경우에는 ***regression(회귀)*** 알고리즘이며, discrete value인 경우에는 ***classification(분류)*** 알고리즘이다. 
- 각 범주별로 사용되는 알고리즘은 다음과 같다. 
  
<img src = 'C:\Users\default.DESKTOP-S5Q9GAA\Documents\Programs\herbwood.github.io\assets\img\mlalg2.png'></img>






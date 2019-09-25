---
layout: post
title:  "Linear Regression"
subtitle:   "how to linear regression"
categories: study
tags: ML
comments: true
---

# 2. Linear Regression

## what?
- 예측하고자 하는 target label y에 대한 독립변수 x가 한 개인 회귀 분석 기법
- y = ax + b 수식으로 표현 가능
- a는 회귀계수, b는 y 절편을 의미한다. 
<img src = 'C:\Users\default.DESKTOP-S5Q9GAA\Documents\Programs\isme2n.github.io\assets\img\linalg.png'></img>

## why?
- 단순하고 직관적이기 때문에 사용하기 쉬움
- 선형관계에 있는 변수의 상관관계를 파악하기 용이함

## why not?
- Model이 단순하여 underfit할 가능성이 높음
- 두 변수 간의 선형관계만을 파악할 수 있어 비선형적인 관계는 알 수 없음
- 이상치에 민감함

## how?
### Input : 
1)Data{($x_i, y_i$)}, M rows(data) and 1 column(feature)
2) Model : $h_\theta(x) =\theta_0 + \theta_1x$
3) Loss function  $J(\theta) = {\frac 1 M}\sum_{i=1}^M (y_i - \hat{y_i})^2$ 
### ```step 1.``` initialize parameters $\theta_0, \theta_1$ for Model 
### ```step 2.``` find optimal paramters
##### 1)  Loss function $J(\theta)$ 계산하기
##### 2) Gradient Descent 방법으로 parameter 최적화 하기(순서 주의!!)
##### $temp0 : = \theta_0 - \alpha$ $\partial J(\theta)\over\partial \theta_0$
##### $temp1 : = \theta_1 - \alpha$ $\partial J(\theta)\over\partial \theta_1$
##### $\theta_0 : = temp0$
##### $\theta_1 : = temp1$
##### 3) Loss function $J(\theta)$가 최소가 될 때까지 반복하기
### ```step 3.``` ouput   $h_\theta(x)$

## Code usage
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# generate dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# train model 
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(y)

# evaluate model accuracy
print(mean_squared_error(y, y_pred))
```
## Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)
[Coursera : Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)



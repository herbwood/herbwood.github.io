---
layout: post
title:  "[ML-04]규제가 있는Linear Regression"
subtitle:   "how to restriced linear regression"
categories: study
tags: ml
comments: true
use_math : true
---

## 4. 규제가 있는 Linear Regression
## 4.1 Lasso Regression

### what?
- Least Absolute Shrinkage Selector Operator의 줄임말
- Linear Regression에 overfit을 방지하기 위한 l1 규제항이 추가된 모델
- 훈련하는 동안에는 규제항이 추가됨(=손실함수에만 추가됨)
- $J_{lasso}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \lambda\sum_{i=1}^N|\theta_i|$
  (M = number of datas, N = number of features)
- 각 feature의 가중치에 대한 규제가 이루어진다. 
- $$\lambda$$가 크면 클수록 규제가 커져 모델이 단순해지며 $$\lambda$$가 작을수록 규제가 작아져 복잡한 모델이 된다. 
![Cap 2019-09-28 15-11-42-712](https://user-images.githubusercontent.com/35513025/65812474-511a4980-e202-11e9-8797-26bd9b1caf06.jpg)

### why?
- l1 규제항 추가로 overfitting을 방지할 수 있다.
- 중요한 몇 개의 변수의 가중치만을 선택하고 나머지 가중치를 0으로 만든다. 즉 데이터에서 중요한 변수가 몇 가지만 있다고 판단하였을 때 사용하기에 적합하다. 이를 feature selection이라고 한다. 
- 일반적으로 모델로 사용되기 보다는 feature selection에 사용된다. 

### why not?
- 자동적으로 중요한 변수를 고르기 때문에 수치적인 중요성은 적으나 데이터를 모델링할 때 중요하거나 흥미로운 변수를 간과할 위험이 있다. 
- 이로 인해 상황에 적합하지 않은 모델을 만들어 낼 수도 있다. 

### how?
#### ```Input```
1)Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)  
2) Model : $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 +...+ \theta_nx_n$
3) Loss function  $$J_{lasso}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \lambda\sum_{i=1}^N|\theta_i|$$   
4) $$\lambda$$ : Loss function 규제항

#### ```Step 1.``` initialize parameters $$\theta_0, \theta_1$$ for Model 
##### 2) Gradient Descent 방법으로 parameter 최적화 하기(순서 주의!!)
##### $temp0 : = \theta_0 - \alpha$ $\partial J(\theta)\over\partial \theta_0$
##### $temp1 : = \theta_1 - \alpha$ $\partial J(\theta)\over\partial \theta_1$
##### (...)
##### $tempn : = \theta_n - \alpha$ $\partial J(\theta)\over\partial \theta_n$
##### $\theta_0 : = temp0$
##### $\theta_1 : = temp1$
##### (...)
##### $\theta_n : = tempn$
##### 3) update된 parameter를 토대로 Loss function 계산
##### 4) Loss function이 최소가 될 때까지 step2의 과정 반복
### ```step 3.``` ouput   l1 규제가 적용된 $$h_\theta(x)$$

### Code usage
```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# generate dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# train model 
model = Lasso(alpha=0.5) # 규제 파라미터 지정해주기
model.fit(X, y)
y_pred = model.predict(y)

# evaluate model accuracy
print(mean_squared_error(y, y_pred))
```

## 4.2 Ridge Regression

### what?
- Linear Regression에 overfit을 방지하기 위한 l2 규제항이 추가된 모델
- 훈련하는 동안에는 규제항이 추가됨(=손실함수에만 추가됨)
- $J_{ridge}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \frac \lambda 2\sum_{i=1}^N\theta_i^2$
  (M = number of datas, N = number of features)
- 각 feature의 가중치에 대한 규제가 이루어진다. 
- $$\lambda$$가 크면 클수록 규제가 커져 모델이 단순해지며 $$\lambda$$가 작을수록 규제가 작아져 복잡한 모델이 된다. 

<img src=![Cap 2019-09-28 15-44-44-713](https://user-images.githubusercontent.com/35513025/65812764-ea4b5f00-e206-11e9-9f25-058d64708400.png)
></img>

### why?
- l2 규제항 추가로 overfitting을 방지할 수 있다.
- 변수의 가중치를 중요도에 따라 전반적으로 줄인다. 중요도가 낮은 변수를 0으로 만들지 않는다.
- 제공된 변수를 중요도만 낮춘 채 전부 다 사용할 수 있다. 

### why not?
- 가중치를 0으로 만들지 않기 때문에 feature selection에 부적합하다. 
- 모든 변수를 사용하기 때문에 데이터의 수를 줄이는 데 적합하지 않다. 

### how?
#### ```Input```
1)Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)  
2) Model : $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 +...+ \theta_nx_n$
3) Loss function  $J_{ridge}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \frac \lambda 2\sum_{i=1}^N\theta_i^2$
4) $$\lambda$$ : Loss function 규제항

#### ```Step 1.``` initialize parameters $$\theta_0, \theta_1$$ for Model 
##### 2) Gradient Descent 방법으로 parameter 최적화 하기(순서 주의!!)
##### $temp0 : = \theta_0 - \alpha$ $\partial J(\theta)\over\partial \theta_0$
##### $temp1 : = \theta_1 - \alpha$ $\partial J(\theta)\over\partial \theta_1$
##### (...)
##### $tempn : = \theta_n - \alpha$ $\partial J(\theta)\over\partial \theta_n$
##### $\theta_0 : = temp0$
##### $\theta_1 : = temp1$
##### (...)
##### $\theta_n : = tempn$
##### 3) update된 parameter를 토대로 Loss function 계산
##### 4) Loss function이 최소가 될 때까지 step2의 과정 반복
### ```step 3.``` ouput   l2 규제가 적용된 $$h_\theta(x)$$

### Code usage
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# generate dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# train model 
model = Ridge(alpha=0.5)
model.fit(X, y)
y_pred = model.predict(y)

# evaluate model accuracy
print(mean_squared_error(y, y_pred))
```

## 4.3 ElasticNet

### what?
- lasso regressio과 ridge regression이 결합된 형태이다. 
-  $J_{elasticnet}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \gamma\lambda\sum_{i=1}^N|\theta_i| + \frac {1-\gamma} 2\lambda \sum_{i=1}^N\theta_i^2$
- $$\gamma$$ 파라미터를 통해 l1 규제와 l2 규제를 조정할 수 있다. 
- $$\gamma = 0$$, -> elasticnet = ridge, $$\gamma = 1$$, -> elasticnet = ridge

### why?
- 일반적으로 변수의 수가 많을 때 사용된다.
- $$\gamma$$를 통해 규제를 조정하여 유연성이 높은 모델을 만들 수 있다.

### why not?
- 연산량이 많아 최적의 모델을 산출하기까지 다른 모델에 비해 시간이 오래 걸린다.
- 유연성이 높은만큼 overfitting될 위험이 있다. 

### how?
#### ```Input```
1)Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)  
2) Model : $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 +...+ \theta_nx_n$
3) Loss function  $J_{elasticnet}(\theta) = {\frac 1 {2m}} \sum_{i=1}^M (h_\theta(x)_i - y_i)^2 + \gamma\lambda\sum_{i=1}^N|\theta_i| + \frac {1-\gamma} 2\lambda \sum_{i=1}^N\theta_i^2$ (h_\theta(x)_i - y_i)^2 + \frac \lambda 2\sum_{i=1}^N\theta_i^2$
4) $$\lambda$$ : Loss function 규제항
5) $$\gamma$$ : l1과 l2 규제항 조정 파라미터

#### ```Step 1.``` initialize parameters $$\theta_0, \theta_1$$ for Model 
##### 2) Gradient Descent 방법으로 parameter 최적화 하기(순서 주의!!)
##### $temp0 : = \theta_0 - \alpha$ $\partial J(\theta)\over\partial \theta_0$
##### $temp1 : = \theta_1 - \alpha$ $\partial J(\theta)\over\partial \theta_1$
##### (...)
##### $tempn : = \theta_n - \alpha$ $\partial J(\theta)\over\partial \theta_n$
##### $\theta_0 : = temp0$
##### $\theta_1 : = temp1$
##### (...)
##### $\theta_n : = tempn$
##### 3) update된 parameter를 토대로 Loss function 계산
##### 4) Loss function이 최소가 될 때까지 step2의 과정 반복
### ```step 3.``` ouput   l1, l2 규제가 적용된 $$h_\theta(x)$$

### Code usage
```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# generate dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# train model 
model = ElasticNet(alpha=0.5, l1_ratio=0.5)
model.fit(X, y)
y_pred = model.predict(y)

# evaluate model accuracy
print(mean_squared_error(y, y_pred))
```

### Reference
[lasso, ridge elasticnet 설명](https://brunch.co.kr/@itschloe1/11)
[lasso regression의 장단점 설명](https://www.quora.com/What-are-the-pros-and-cons-of-lasso-regression)
[lasso regression과 elasticnet 비교 코드](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)
[ridge regression 장단점 설명](https://stepupanalytics.com/ridge-regression-and-its-application/)
[elasticnet 장단점](https://stats.stackexchange.com/questions/345343/any-disadvantages-of-elastic-net-over-lasso)
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
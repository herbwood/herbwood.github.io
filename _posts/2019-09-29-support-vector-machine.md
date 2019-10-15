---
layout: post
title:  "[ML-07]Support Vector Machine"
subtitle:   "how to support vector machine"
categories: study
tags: ml
comments: false
use_math : true
---
## 7. Support Vector Machine
## 7.1 Support Vector Machine for Classification

### what?
- 서로 다른 범주에 속하는 데이터를 구분짓는 hyper plane과 support vector(범주 최전방에 속해있는 데이터) 간의 거리를 최대화하는 분류 알고리즘. Large Margin Classifier라고도 불림.
![1_QJZVKh-YHhPn5Q83kzJ96Q](https://user-images.githubusercontent.com/35513025/65860919-3f71a700-e3a6-11e9-946d-5c7d6c7bb06a.png)

1) Logistic Regression과의 차이점
- logistic regression Loss function
$$\frac 1 m [\sum_{i=1}^M(y_ilog(h_\theta(x_i))+ (1-y_i)log(1-h_\theta(x_i)))] + \frac {\lambda} {2m}\sum_{j=0}^N\theta_j^2$$  
if $$\theta^Tx \ge 0$$, then $$y=1$$, else $$y=0$$  

- Support Vector Machine Loss function
$$C\sum_{i=1}^M(y_icost_1(h(x_i)) + (1-y_i)cost_0(1-h(x_i))) + \frac 1 2 \sum_{j=0}^N\theta_j^2$$
if $$\theta^Tx \ge 1$$, then $$y=1$$, else $$y=0$$  
- support vector machine은 기존 logistic regresion보다 margin을 더 크게 만든다.
- 왼쪽항이 규제항으로 바뀐다.
logistic regression : A + $$\lambda$$B
support vecotr machine : CA + B(C=$$\frac 1 {\lambda}$$)

2) Mathmetics behind Support Vector Machine
- 규제항을 제외한 $$\frac 1 2 \sum_{j=0}^N\theta_j^2$$을 minimize해야함
#####= $$\frac 1 2 (\theta_0^2 + \theta_1^2 + ... + \theta_n^2)$$
##### ex) n = 2, $$\frac 1 2 (\theta_0^2 + \theta_1^2)$$
![Cap 2019-10-01 18-07-45-526](https://user-images.githubusercontent.com/35513025/65949132-95158480-e476-11e9-8aed-e623aed1d4ba.png)
##### $$\theta^Tx = p * ||\theta|| \ge 1$$이 성립하기 위해서 SVM 알고리즘은 p의 크기를 최대화함으로써 $$\theta$$를 minimize할 것이다. 

3) Non-linear classification

3-1) polynominal features를 더해줌으로써 가능
ex) $$x_1$ -> $$x_1^2 + 2$$

3-2) similarity(유사도) 추가
- 각 데이터를 랜드마크로 지정해 랜드마크와 데이터 사이의 거리를 feature로 추가하는 방식
- $$l$$ = landmark, $$f_1$$ = similarity($$x, l$$)
- similarity를 정하는 방식에는 여러 가지가 있다. 그 중 ```Gaussian kernel```은 다음과 같다.   
Gaussian kernel = $$exp(\frac {-||x - l||^2} {2\sigma^2})$$
(if $$x \approx l$$ : $$exp(\frac {-0} {2\sigma^2}) = 1$$,
else $$exp(\frac {-large} {2\sigma^2})$$)  
$$\theta_0 + \theta_1x_1 + ... + \theta_nx_n$ -> $\theta_0 + \theta_1f_1 + ... + \theta_nf_n$$

### why?
- non-linear 한 데이터 분류가 가능하다. 
- feature의 수가 data의 수보다 많을 때 효과적이다. 
- 고차원 데이터에 대해서 좋은 성능을 보인다. 
- decision boundary는 support vector의 영향만을 받기 때문에 이상치의 영향을 조금 받는다. 

### why not?
- 연산량이 많아 데이터의 크기가 클 경우 결과를 내기까지 시간이 오래걸릴 수 있다. 
- 범주가 겹치는 경우 좋은 성능을 보이지 않는다. 

### how?
#### ```Input``` 
1)Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)  
2) Model : $$h_\theta(x) =\theta_0 + \theta_1x + ... + \theta_nx_n$$
3) Loss function  $$C\sum_{i=1}^M(y_icost_1(h(x_i)) + (1-y_i)cost_0(1-h(x_i))) + \frac 1 2 \sum_{j=0}^N\theta_j^2$$ 
4) parameters C, $$\gamma$$(if using gaussian kernel), 
#### ```Step 1``` initialize parameters $$\theta_0, \theta_1,..., \theta_n$$ for Model 
#### ```step 2.``` find optimal paramters
- 1)  Loss function $$J(\theta)$$ 계산하기
- 2) Gradient Descent 방법으로 parameter 최적화 하기(순서 주의!!)
##### $$temp0 : = \theta_0 - \alpha$ $\frac {\partial J(\theta)} {\partial \theta_0}$$
##### $$temp1 : = \theta_1 - \alpha$ $\frac {\partial J(\theta)} {\partial \theta_1}$$
##### $$tempn : = \theta_n - \alpha$ $\frac {\partial J(\theta)} {\partial \theta_n}$$
##### $$\theta_0 : = temp0$$
##### $$\theta_1 : = temp1$$
##### $$\theta_n : = tempn$$
- 3) update된 parameter를 토대로 Loss function 계산
- 4) Loss function이 최소가 될 때까지 step2의 과정 반복
#### ```step 3.``` Ouput : optimal hyper plane $h_\theta(x)$

### Code usage

1) Linear Classification
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data'][:, (2,3)]
y = iris['target']

setosa_or_versicolor = (y==0) | (y==1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel = 'linear', C=float('inf'))
svm_clf.fit(X, y)
```

2) Non-linear Classification by adding polynominal features
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)
```
3) Non-linear Classification by adding Gaussian similarity features
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```

## 7.2 Support Vector Machine for Regression

### what?
- 범주를 구분짓는 Hyper plane과 support vector를 지나면서 hyper plane과 평행한 boundary line 사이의 공간에 데이터가 최대한 많이 속하도록 학습시키는 알고리즘
- epsilon 파라미터가 hyper plane과 boundart line 사이의 거리를 조정한다. epsilon이 커질수록 포함되는 데이터의 수가 많아진다. 
![Cap 2019-10-02 16-32-44-364](https://user-images.githubusercontent.com/35513025/66025804-4b8a6f80-e532-11e9-98ac-5df5847a33b6.jpg)

### why?
- Support Vector Machine for Classification과 동일

### why not?
- Support Vector Machine for Classification과 동일

### how?
- Support Vector Machine for Classification과 동일

### Code usage
```python
import numpy as np
import sklearn.svm import SVR

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()

svm_poly_reg = SVR(kernel="poly", gamma='auto', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```

### Tips
#### Support Vector Machine Parameters
  
  
|parameter|의미|higher|lower|  
|:---------:|:---:|:-------:|:------:|  
|C|얼마나 많은 샘플이 다른 범주에 놓일지 결정|이상치 가능성을 높게 봄, 높으면 underfit, hard margin|이상치 가능성을 낮게 봄, 낮으면 overfit, soft margin|
|gamma|하나의 데이터 샘플의 영향력을 결정|작은 표준편차, 영향력 거리가 작음, underfit|큰 표준편차, 영향력 거리가 큼, overrfit|
|epsilon|마진 안에 얼마나 많은 샘플이 들어올지 결정|샘플이 마진 안에 들어올 수 있는 범위가 넓어짐, underfit|샘플이 마진 안에 들어올 수 있는 범위가 좁아짐, overfit|
  
  




### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Coursera : Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)  
[Support Vector Machine 장단점](https://data-flair.training/blogs/svm-support-vector-machine-tutorial/)  
[Support Vector Machine for Regression 설명](https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff)  



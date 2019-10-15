---
layout: post
title:  "[ML-13]Gradient Boosting"
subtitle:   "how to gradient boosting"
categories: study
tags: ml
comments: false
use_math : true
---

## 13.Gradient Boosting

### what?
- 매우 간단한 모델을 구축한 후, Residual에 Fitting한 모델을 만들고, 이 두 개를 결합한다. 그리고 결합된 모델에서 다시 Residual이 나오면 다시 이 Residual에 Fitting하는 모델을 만들어나가고 이를 계속 반복하여 최종 모델을 만드는 알고리즘
- Adaboost와 다르게 매 interation마다 만드는 tree의 크기가 더 크다. 일반적으로 8~32개 정도의 leaf node를 만든다.

![Cap 2019-10-03 21-05-03-909](https://user-images.githubusercontent.com/35513025/66125267-7b647080-e621-11e9-8a24-b5386cb8cf0d.jpg)

### why?
- 여러 타입의 데이터를 잘 처리한다
- 일반적으로 성능이 준수하게 나온다
- 이상치에 강건(robust)하다

### why not?
- 부스팅 알고리즘의 특성상 계속 약점(오분류/잔차)을 보완하려고 하기 때문에 잘못된 레이블링이나 아웃라이어에 필요 이상으로 민감할 수 있다.
- 순차적으로 트리를 학습시키기 때문에 학습 속도가 느리다

### 13.1 Gradient Boosting for Regression
### how?
#### ```Input``` 
1)Data($$x_i, y_i$$)  
2) diffrentiable Loss function : $$L(y_i, \gamma) = \frac 1 2 \sum_{i=1}^n(y_i - \hat{y})^2$$

#### ```Step 1``` Initialize model with a constant value   
$$F_0(x) = \underset \gamma {argmin} \sum_{i=1}^nL(y_i, \gamma)$$  
- $$\gamma$$ : predicted value  
- $$F_0(x)$$ : initial predicted value(=average of all target values)  

#### ```Step 2``` for m = 1 to M(tree number 1 to M)
1) Compute $$r_{im} = - [\frac {\partial L(y_i, F(x_i))} {\partial F(x_i)}]_{F(x)=F_{m-1}(x)}$$ for i = 1,...,n
- $r_{im}$ 계산 결과 $$(y_i - \hat(y_i))$$라는 residual이 나온다. 이를 pseudo residual이라고 한다. 
- 모든 샘플에 대하여 residual을 구한다.
- denote
$r$ : residual  
$i$ : sample number  
$n$ : max sample number
$m$ : tree index

2) Fit a regression tree to the $r_{im}$ values and create terminal regions $$R_{jm}$$, for j = 1,...,$J_m$  
- target label이 아닌 residual을 예측하기 위한 결정 트리를 만든다.  
- denote
$$j$$ : leaf node index  
$$R_{jm}$$ : m번째 트리의 j번째 leaf node

3) For j = 1,...,$$J_m$$ compute $$r_{jm} = \underset \gamma {argmin} \underset {x_i \in R_{ij}}{\sum} L(y_i, F_{m-1}(x_i) + \gamma)$$  
- 각 leaf node마다 residual에 대한 output을 계산  
- $$r_{im}$$ : leaf node에 속한 샘플의 평균  

4) Update $$F_M(x) = F_{m-1}(x) + \nu\sum_{j=1}^{J_m}r_{jm}I(x\in R_{jm})$$  
- predict target label for each samples
- denote
$$F_{m-1}(x)$$ : last prediction  
$$\nu$$ : learning rate
$$r_{jm}I(x\in R_{jm})$$ : sample x가 있는 leaf node의 $$r_{jm}$$(predicted residual)
- 작은 learning rate는 각각의 트리가 전체 예측에 미치는 영향력을 줄여준다. 이를 통해 gradient boosting은 장기적으로 높은 정확도를 가지게 된다. 

#### ```Step 3``` output $$F_M(x)$$

### Code usage
```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
gbrt.fit(X, y)
```

### 13.2 Gradient Boosting for Classification
### how?
#### ```Input``` 
1)Data{($$x_i, y_i$$)}  
2) diffrentiable Loss function : $$-\sum_{i=1}^N[y_ilog(p) + (1-y_i)log(1-p)]$$   
Loss function을 정리하면 $$-Observed * log(odds) +log(1 + e^{log(odds)})$$

- deonote 
$$y_i$$ : Observed value  
$$p$$ : predicted probability(=$$\frac {e^{log(odd)}} {1+e^{log(odd)}}$$)
$$log(odd)$$ : $$log(\frac {yes} {no})$$

#### ```Step 1``` Initialize model with constant value :  
$$F_0(x) = \underset \gamma {argmin} \sum_{i=1}^N L(y_i, \gamma)$$  
- Initial prediction $$F_0(x)$$를 구한다. 
- predictied probability인 $$p$$를 찾고 $$log(odds)$$를 구한다. 


#### ```Step 2``` for m = 1 to M(tree number 1 to M)
1) Compute $$r_{im} = - [\frac {\partial L(y_i, F(x_i))} {\partial F(x_i)}]_{F(x)=F_{m-1}(x)}$$ for i = 1,...,n  
- 각 sample에 대한 residual을 구한다  

2) Fit a regression tree to the $$r_{im}$$ values and create terminal regions $$R_{jm}$$, for j = 1,...,$$J_m$$  
- target label이 아닌 residual을 예측하기 위한 결정 트리를 만든다. 

3) For j = 1,...,$$J_m$$ compute $$r_{jm} = \underset \gamma {argmin} \underset {x_i \in R_{ij}}{\sum} L(y_i, F_{m-1}(x_i) + \gamma)$$   
- 각 leaf node마다 residual에 대한 output을 계산 
- $$\gamma = \frac {Residual} {p(1-p)}$$

4) Update $$F_M(x) = F_{m-1}(x) + \nu\sum_{j=1}^{J_m}r_{jm}I(x\in R_{jm})$$  
- predict target label(log(odds)) for each samples

5) find p for each value by using log(odds)
- $$p = \frac {e^{log(odd)}} {1+e^{log(odd)}}$$

#### ```Step 3``` output $$F_M(x)$$

### Code usage
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

### Tips : Gradient Boosting Parameters
- learning rate : 각각의 트리가 전체의 prediction에 어느 정도 영향력을 줄지 결정
- n_estimators : 몇 개의 트리를 사용할지 결정 
- max_depth : 트리의 최대 깊이. 

### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Gradient Boosting에 대한 직관적 설명](https://4four.us/article/2017/05/gradient-boosting-simply)  
[그저 갓갓 statquest](https://www.youtube.com/watch?v=2xudPOBz-vs)  
[pros and cons of Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)  




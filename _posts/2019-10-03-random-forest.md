---
layout: post
title:  "[ML-11]Random Forest"
subtitle:   "how to random forest"
categories: study
tags: ml
comments: false
use_math : true
---

## 11. Random Forest

### what?
- 훈련 과정에서 구성한 다수의 결정 트리로부터 부류(분류) 또는 평균 예측치(회귀 분석)를 출력
- regression과 classification 모두 사용 가능
- Bagging을 통해 데이터를 복원 추출하는 방식을 사용하는 동시에 Random Subspace를 통해 feature를 랜덤하게 추출하여 사용한다. 이렇게 추출된 데이터와 특징을 기반으로 결정 트리를 만들고, 여러 결정 트리의 예측치를 종합함으로써 결과를 내는 방식으로 동작한다. 즉 ***Random Forest = Bagging + Random Subspace + Decision Tree***
![Cap 2019-10-03 07-50-32-739](https://user-images.githubusercontent.com/35513025/66087475-8df39100-e5b2-11e9-8e30-b409e3f52f61.jpg)

- 결정 트리는 학습데이터에 따라 생성되는 결정트리가 매우 달라지기 때문에 일반화하여 사용하는데 어려움이 있다. 또한 계층적 접근 방식이기 때문에 중간의 에러를 다음 단계로 전파하는 문제가 있다. 하지만 Random Forest에서는 이러한 문제를 데이터와 특성을 sampling 하는 방식을 통해 해결하였다. 
- Random Forest는 결정트리마다 다르게 주어진 데이터와 특성으로 인해 임의성을 가지게 된다. 이로 인해 각 트리의 상관관계가 약해져 오히려 일반화 성능이 향상되며, 노이즈가 포함된 데이터에 대해서도 robust해진다. 
- 각 트리의 훈련 과정에서 특정 feature나 일부 feature가 주로 선택된다면 target label을 예측하는데 있어 중요한 feature를 찾을 수 있다. 
- OOB 방식을 통해 평가한다. 


### why?
- 결정트리의 문제점인 일반화 성능을 향상시킨 동시에 노이즈에 강건하다. 또한 결정 트리보다 overfitting될 가능성이 낮다. 
- 자동적으로 missing value를 처리한다(proximity matrix를 기반으로 한 weight average를 통해 가능함).
- feature scaling을 필요로 하지 않는다(거리 기반 연산이 아닌 규칙 기반 연산에 근거하기 때문에).

### why not?
- 연산량이 많기 때문에 결과를 내기까지 시간이 오래 걸린다.
- 마찬가지로 많은 용량을 차지한다.
- 일반적인 모델보다 해석하기 더 어렵다. 

### how?
#### ```Input``` 
1) Data{($x_i, y_i$)}, M rows(data) and N columns(feature)  
2) n_estimators, max_depth 등등 트리의 크기 결정 파라미터
#### ```Step 1``` bagging과 random supspace를 통한 데이터와 feature 추출
#### ```Step 2```
1) n_estimaotors만큼의 결정 트리 생성 후 학습
2) voting을 통한 예측치 결정 트리의 예측치 취합
#### ```Step 3``` output : predictions on new instance

### Code usage
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
```
### Tips : Random Forest parameters
- n_estimators : 생성할 트리의 개수. 너무 많으면 overfitting
- max_depth : 트리의 최대 깊이. 너무 높으면 overfitting 발생
- min_samples_split : 분기하기 위한 최소 데이터의 수. 너무 낮으면 overfitting 발생
- min_samples_leaf : terminal leaf에 속할 최소 샘플 수. 너무 높으면 overfitting 발생
- max_features : 사용할 최대 feature 수. 너무 높으면 overfitting 발생
- max_leaf nodes : terminal leaf의 최대 수. 너무 높으면 overfitting 발생


### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Random Forest Wiki](https://ko.wikipedia.org/wiki/랜덤_포레스트)  
[Random Forest 쉽게 풀이](https://gentlej90.tistory.com/37)  
[Random Forest 장단점](http://theprofessionalspoint.blogspot.com/2019/02/advantages-and-disadvantages-of-random.html)  
[Random Forest가 missing value를 채우는 방법](https://www.youtube.com/watch?v=nyxTdL_4Q-Q)


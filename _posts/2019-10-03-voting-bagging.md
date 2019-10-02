---
layout: post
title:  "[ML-10]Voting and Bagging"
subtitle:   "how to Voting and Bagging"
categories: study
tags: ml
comments: false
use_math : true
---

## 10. Voting and Bagging
## 10.1 Voting

### what?
- 여러 개의 분류기를 사용하여 결과를 종합하는 알고리즘
- regression과 classification 모두 가능하다. 
- regression은 각 분류기의 예측값의 평균을 사용한다.
- classification은 각 분류기의 최빈값을 사용하는 hard voting과 각 분류기의 범주에 대한 예측 확률의 평균을 사용하는 soft voting이 있다.(soft voting이 hard voting보다 좋은 성능을 보인다고 한다)
![Cap 2019-10-03 03-39-59-989](https://user-images.githubusercontent.com/35513025/66072158-7f47b280-e58f-11e9-814b-4edbae3a5602.jpg)

### why?
- 누적 확률로 인해 voting 알고리즘을 사용할 경우 정확도가 단일 모델을 사용할 때보다 더 높아진다. 

### why not?
- 각 모델이 독립적이어야 하고 오차에 대한 상관관계가 없어야 한다. 

### how?
#### ```Input```
1) Data{($x_i, y_i$)}, M rows(data) and N columns(feature)  
2) Algorithms to use
#### ```Step 1``` voting에 사용할 알고리즘을 각각 훈련시키기
#### ```Step 2``` voting 방법에 따라 결과 종합하기
#### ```Step 3``` output : optimal value for new instance

### Code usage
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression(solver='liblinear', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma='auto', random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)
```


## 10.2 Bagging

### what?
- 하나의 모델에 대해 중복을 허용한 서로 다른 데이터셋을 사용하여 학습시키는 알고리즘
- Bagging은 Boostrap Aggregating의 줄임말이다.(Bootstrap은 통계학에서 중복을 허용한 복원 추출을 의미한다)
- regression과 classification에 모두 사용 가능
![Cap 2019-10-03 04-00-40-987](https://user-images.githubusercontent.com/35513025/66073619-6391db80-e592-11e9-8d9b-ce29597e0451.jpg)

1) Pasting 
- Pasting은 비복원 추출을 의미한다
- bootstrap 하이퍼 파라미터로 비복원 추출 여부를 지정할 수 있다. 
- 일반적으로 bagging이 pasting보다 성능이 더 높게 나온다.

2) OOB(Out Of Bag)
- Bagging 알고리즘은 복원 추출 방식이기 때문에, 어떤 데이터는 학습에 여러 번 사용되는 반면, 어떤 데이터는 학습에 전혀 사용되지 않는다.
- m개의 샘플에서 하나의 샘플을 추출한다고 했을 때 특정 샘플이 추출되지 않을 확률은
$(1- \frac 1 m)$이며 이를 m번 추출한다고 했을 때 확률 $(1- \frac 1 m) ^ m$은 0.37에 수렴한다. 즉 37%의 데이터는 학습에 사용되지 않는다.
- oob_score 하이퍼 파라미터 지정을 통해 사용되지 않은 데이터를 validation용 데이터를 사용할 수 있도록 설정할 수 있다. 

3) Random Patch and Random Subspace
- 특성과 데이터에 대한 sampling 여부를 결정하는 방식
- Random Patch : 특성 및 데이터 모두 복원 추출하는 방식
***bootstrap = True, bootstrap_features = True***
- Random Subspace : 특성에 대해서만 복원 추출을 적용하고 데이터는 전체 다 사용하는 비복원추출 방식을 사용함
***bootstrap = False, bootstrap_features = True*** 

### why?
- 단일 모델을 사용해 학습하는 경우보다 편향은 비슷하지만 분산이 줄어든다. 

### why not?


### how?
#### ```Input```
1) Data{($x_i, y_i$)}, M rows(data) and N columns(feature)  
2) Algorithms to use
3) Boostrap methods 
#### ```Step 1``` 데이터 및 feature 샘플링
#### ```Step 2``` 추출된 데이터 및 feature에 대한 학습 진행
#### ```Step 3``` output : optimal value for new instance

### Code usage
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```
### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      	
[voting classifier 이미지](http://coursepress.lnu.se/kurs/applied-machine-learning/files/2018/09/8.Ensemble-Learning1.pdf)  
[앙상블 모델 설명](https://excelsior-cjh.tistory.com/166)




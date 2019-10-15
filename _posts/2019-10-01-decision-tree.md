---
layout: post
title:  "[ML-09]Decision Tree"
subtitle:   "how to Decision Tree"
categories: study
tags: ml
comments: false
use_math : true
---

## 9. Decision Tree

### what?
- feature에 따라 데이터를 순차적으로 분리하는 알고리즘
- regression과 classification에 모두 사용 가능하다. 
- 불확실성, 분순도가 감소하는 방향으로, 데이터를 분리한다. Node로 선택했을 때 불순도가 가장 낮은 feature을 우선적으로 선택하여 분리를 진행함. 
- 불순도를 계산하는 방식에 따라 CART(Classification And Regression Tree) 알고리즘과, ID3 알고리즘으로 나눠진다. 
- 불순도 계산으로 데이터를 분기한 후 overfitting을 방지하기 위해 가지치기(pruning) 작업을 실시한다. 
- Loss function : $$CC(T) = Err(T) + \alpha * L(T)$$  
$$Err(T)$$ : 데이터에 대한 오분류율
$$L(T)$$ : terminal node(leaf node)의 수
![Cap 2019-10-03 01-01-57-092](https://user-images.githubusercontent.com/35513025/66061213-fa9e6980-e579-11e9-87b6-80f37d71a8d8.jpg)

예시) Chest Pain 여부, diabetes 여부로 heart disease 예측하기
![Cap 2019-10-03 01-17-39-020](https://user-images.githubusercontent.com/35513025/66069383-393c2000-e58a-11e9-8d52-6142543a476f.png)

1) CART Algorithm
- 지니 계수(Gini Index)를 활용하여 불순도를 계산함. 지니 계수는 개별 node의 불순도인 지니 불순도(Gini Impurity)를 통해 얻을 수 있다. 
- Gini Impurity : $$G_i = 1 - \sum_{k=1}^N p_{i,k}^2$$

- Gini Index : $$G(A) = \sum_{i=1}^d R_i * G_i$$
- 
$$i$$ : node의 번호.   
$$k$$ : target label에 대한 여부.   
$$p_{i,k}$$ :  i번째 node의 k 대답에 대한 발생 비율.   
$$d$$ : 나눈 node의 수(여기서는 yes, no로 이진분류를 했으므로 $$d=2$$)  
$$R_i$$ : 전체 데이터 중 i번째 노드의 비율  
- 예를 들어 위의 예시에서 Chest Pain에 대한 $$p_11$$ 변수는 Chest Pain이 있으며 Heart Disease도 있을 확률이다.

- 위의 예시의 경우, Chest Pain feature의 지니 불순도와 지니 계수는  $$G_1 = 1 - [(\frac {105} {105 + 39}) ^ 2 + (\frac {39} {105 + 39}) ^ 2] = 0.395$$  
$$G_2 = 1 - [(\frac {34} {34 + 125}) ^ 2 + (\frac {125} {34 + 125}) ^ 2] = 0.336$$   
$$G(Chest Pain) = \frac {144} {144 + 159} * G_1 + \frac {159} {14 + 159} * G_2= 0.364$$  
- 같은 방법으로 $$G(Diabetes) = 0.355$$
- Diabtes feautre가 Chest Pain보다 지니계수가 더 낮으므로 처음 데이터를 분리할 때 Diabetes feature를 사용한다. 

2) ID3 Algorithm
- Information Gain을 통해 불순도를 계산함. Information Gain은 엔트로피가(Entropy)를 통해 얻을 수 있다. Information Gain이 가장 높은 feature를 분기하는 데 사용함
- Entropy : $$E = -p(+)log(p(+)) - p(-)log(p(-))$$  
p(+) : target label에 속하는 데이터의 비율
p(-) : target label에 속하지 않는 데이터의 비율
- Information Gain : $$I = E_{target} - E_{feature}$$

3) for Regression
- 분기로 인해 나눠진 node에 속한 데이터의 평균값으로 예측한다


### why?
- 직관적이고 이해하기 쉽다. 
- 중요한 feature에 대해 파악할 수 있다.
- 이상치에 큰 영향을 받지 않아 preprocessing이 다른 알고리즘에 비해 상대적으로 덜 필요하다.
- 비모수적 방법이기 때문에 데이터에 대한 가정이 없다.

### why not?
- 과적합의 가능성이 높음
- decision boundary를 x,y 축에 수직으로만 형성할 수밖에 없다. 
- 데이터의 회전에 민감하여 일반화하기 힘들다.

### how?
#### ```Input``` 
1) Data$${(x_i, y_i)}$$, M rows(data) and N columns(feature)  
2) $$\alpha$$ : L(T)를 반영하는 정도(0.01~0.1 사이의 값 사용)  
3) Loss function : $$CC(T) = Err(T) + \alpha * L(T)$$  
4) max_leaf_node : terminal node의 수를 정하여 pruning 작업의 정도를 결정함  
5) 불순도 계산 알고리즘   
#### ```Step 1``` 모든 feature에 대한 불순도 계산 후 불순도가 가장 낮은 순으로 분기
#### ```Step 2``` 적절한 pruning 작업 실행
#### ```Step 3``` output : optimal Tree

### Code usage
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
```

### Tips : Decision Tree parameters
- max_depth : 트리의 최대 깊이. 너무 높으면 overfitting 발생
- min_samples_split : 분기하기 위한 최소 데이터의 수. 너무 낮으면 overfitting 발생
- min_samples_leaf : terminal leaf의 최소 수. 너무 높으면 overfitting 발생


### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Decision Tree 설명](https://ratsgo.github.io/machine%20learning/2017/03/26/tree/)  
[Decision Tree 장단점 설명](https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb)  
[ID3 알고리즘 설명 유튜브 영상](https://www.youtube.com/watch?v=n0p0120Gxqk)  



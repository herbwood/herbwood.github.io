---
layout: post
title:  "[ML-06]Naive Bayes"
subtitle:   "how to naive bayes"
categories: study
tags: ml
comments: false
use_math : true
---

## 6. Naive Bayes

### what?
- 베이즈 이론을 토대로 데이터를 분류하는 알고리즘. 
- Bayesian Theorem
$$P(A|B) = \frac {P(B \cap A)} {P(B)} = \frac {P(B|A)P(A)} {\sum {P(B|A)P(A)}}$$
($$P(1|x) = x$$가 범주 1에 속할 확률, $$P(0|x) = x$$가 범주 0에 속할 확률로 이해하면 쉬움)
- 분모가 0이 되는 경우를 방지하기 위해 상수 k를 더해주는 smoothing 기법을 도입한다. 
- 각 범주에 속할 확률을 구한 후 가장 확률이 높은 범주로 분류하는 방식

### why?
- 쉽고 빠르게 실행할 수 있으며 메모리 측면에서도 효율적이다. 
- 노이즈와 결측 데이터가 있어도 잘 수행한다.
- 훈련할 때 상대적으로 적은 데이터만 있어도 좋은 성능을 보인다.

### why not?
- 모든 데이터의 중요도가 같고 서로 독립적이라는 가정에 근거한다. 이러한 가정에 맞지 않는 데이터의 경우 적용하기 힘들다. 
- 수치형 데이터에 적합하지 않다. 

### how?
#### ```Input``` 
1)Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)    
2) Model : $$P(A|B) = \frac {P(B|A)P(A) + k} {\sum {P(B|A)P(A)} + 2k}$$ 
#### ```step 1.``` $$P(x), P(1|x), P(2|x)$$ 확률 구하기
#### ```step 2.``` $$P(1|x), P(2|x)$$ 구하기
##### 1) 베이즈 정리 도입
##### $$P(1|x) = \frac {P(1)P(x|1)} {P(x)}$$
##### $$P(2|x) = \frac {P(2)P(x|2)} {P(x)}$$
##### 두 확률 모두 $$\frac 1 {P(x)}$$가 공통적으로 있으므로 생략 가능
##### $$P(1|x) =  {P(1)P(x|1)} $$
##### $$P(2|x) =  {P(2)P(x|2)} $$
##### 2) 두 확률의 대소 비교하여 범주 $$y_1$$ 결정 
##### if $$P(1|x) \ge 0.5$$, then $$y_1 = 1$$
##### if $$P(2|x) < 0.5$$, then $$y_1 = 2$$
#### ```step 3.``` ouput   데이터에 대한 범주값 $$y_i$$

### Code usage
```python
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# load data
iris = datasets.load_iris()

# model selection
gnb = GaussianNB()
gnb.fit(iris.data, iris.target)
y_pred = gnb.predict(iris.data)

# evaluation
print("Number of mislabeled points out of a total %d points : %d"
       % (iris.data.shape[0],(iris.target != y_pred).sum()))
```
## Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Coursera : Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)    
[Naive Bayes 예시](https://gomguard.tistory.com/69)    
[Naive Bayes 장단점](http://w3devlabs.net/wp/?p=17273)    



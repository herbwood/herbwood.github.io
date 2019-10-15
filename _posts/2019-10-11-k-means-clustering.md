---
layout: post
title:  "[ML-19]K-Means Clustering"
subtitle:   "how to k-means clustering"
categories: study
tags: ml
comments: false
use_math : true
---

## 19. K-Means Clustering

### what?
- k-평균 알고리즘(K-means algorithm)은 주어진 데이터를 k개의 클러스터(군집)로 묶는 비지도학습 알고리즘으로, 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작한다
- K-Means Clustering은 EM(Expectation-Maximization) 알고리즘을 기반으로 작동한다

![kmc](https://miro.medium.com/max/1526/1*vNng_oOsNRHKrlh3pjSAyA.png)

### why?
- 분류에 대한 label이 없는 경우 데이터를 분류하는 데 효과적이다 

### why not?
- 처음 정해진 초기값의 위치에 따라 결과가 다르게 나타난다
- 군집의 크기나 밀도가 다른 경우 제대로 작동하지 않는다
- 데이터 분포가 특이한 경우에도 좋은 효과를 보이지 못한다

### how?
#### ```Input``` 
1) Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)  
2) k : 클러스터의 수   
#### ```step 1.``` k개의 centroid(클러스터의 중심)를 sample에서 임의로 추출하여 초깃값을 설정한다
- $$\mu_i$$ : $$i$$번째 centroid의 위치

#### ```step 2.``` EM algorithm
1) Expectation step : 각 데이터 오브젝트들에 대해 k 개의 클러스터 중심과의 거리를 구하고, centroid와의 거리를 기반으로 데이터에 클러스터를 할당한다.   
2) Maximization step : Expectation step에서 할당된 데이터를 기준으로 centroid를 다시 계산한다.  
3) 각 데이터의 소속 클러스터가 바뀌지 않을 때 까지 2, 3 과정을 반복한다.  
#### ```step 3.``` 데이터에 할당된 k개의 클러스터


### Code usage
```python
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples = 300, centers = 4,
                      cluster_std = 0.60, random_state = 0)
plt.scatter(X[:, 0], X[:, 1], s = 50)

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
```
### Reference 
[핸즈온 머신러닝](https://github.com/rickiepark/handson-ml)      
[Coursera : Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)  
[K-Means Clustering wiki](https://ko.wikipedia.org/wiki/K-%ED%8F%89%EA%B7%A0_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)  



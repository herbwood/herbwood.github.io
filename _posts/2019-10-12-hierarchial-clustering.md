---
layout: post
title:  "[ML-20]Hierarchial Clustering"
subtitle:   "how to hierarchial clustering"
categories: study
tags: ml
comments: true
use_math : true
---

## 20. Hierarchial Clustering

### what?
- 계층적 트리 모형을 이용해 개별 개체들을 순차적, 계층적으로 유사한 개체 내지 그룹과 통합하여 군집화를 수행하는 알고리즘
- Hierarchial Clustering을 통해 형성된 dendrogram에서 적정한 수준에서 끊어줌으로써 계층의 수를 정할 수 있다. 

![hierarchial clustering](http://www.sthda.com/sthda/RDoc/figure/clustering/hierarchical-k-means-clustering-hierarchical-clustering-1.png)

### why?
- K Means Clustering과는 다르게 군집의 수를 미리 정해주지 않아도 된다
- 직관적이며 사용하기 쉽다

### why not?
- 거리와 유사도를 기반으로 한 결정은 이론적 기반이 약해 좋은 결과를 내지 못할 때가 많다
- 연산량이 많아 학습하기까지 K-Means Clustering보다 오래 걸린다

### how?
#### ```Input``` 
1) Data{($$x_i, y_i$$)}, M rows(data) and 1 column(feature)    
#### ```step 1.```  데이터 간의 유사도(similarity)나 거리(distance) matrix
![distance matrix](http://i.imgur.com/25IT5fI.png)
#### ```step 2.``` sample 및 군집 간의 유사도 계산
1) 유사도가 가장 같은 sample끼리 묶어 군집으로 만들
2) 데이터와 군집 간의 유사도를 계산
3) 분석 대상 관측치가 없을 때까지 1),2) 과정 반복
#### ```step 3.``` 분기에 따른 클러스터를 가진 sample


### Code usage
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])
    
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
```
### Reference 
[Hierarchial Clustering 설명](https://ratsgo.github.io/machine%20learning/2017/04/18/HC/)  



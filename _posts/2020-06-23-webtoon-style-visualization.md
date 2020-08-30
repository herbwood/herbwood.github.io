---
layout: post
title:  "[Project]웹툰 그림체 시각화 프로젝트 Webtoon Style Visualization"
subtitle:   "webtoonstylevisualization"
categories: study
tags: project
comments: true
use_math : true
---

이번 포스팅에서는 웹툰 그림체 분류 프로젝트 **Webtoon Style Visualization** 프로젝트에 대해 소개하도록 하겠습니다. 저는 웹툰에 딥러닝을 적용하여 유의미한 결과를 얻어내는 프로젝트에 대해 생각했습니다. 그 중에서 웹툰별로 그림체를 분류할 수 있다면 인기 웹툰의 그림체에서 나타나는 공통점을 찾을 수 있겠다고 생각하여 웹툰 그림체 분류 프로젝트를 시작하게 되었습니다. 

## 프로젝트 개요

웹툰의 그림체를 분류하는 문제는 일반적인 이미지 분류 문제와 크게 다르게 때문에 구현 방식에 대해 먼저 고민해보았습니다. 이미지 분류 문제는 이미지 상에 나타나는 동일한 객체를 학습합니다. 반면 그림체 분류 문제는 서로 다른 객체에 대한 공통적인 그림체를  추출하는 것을 목표로 합니다. 이러한 이미지 상에서 나타나는 "그림체"라는 고차원적인 특징을 추출하기 위해서 저는 **Style transfer** 방식을 사용했습니다. 

<br>

## 프로젝트 진행 과정

### 데이터 수집

<p align="center"><img src="https://user-images.githubusercontent.com/35513025/82119422-67890b80-97b9-11ea-9cb6-ecc203b93217.jpg"></p><p align="center">[그림 1] 수집한 웹툰 20 작품</p>

- 데이터 수집을 위해 저는 당시 연재되고 있던 네이버 웹툰 20개의 작품에 대한 이미지를 수집했습니다. 요일별, 장르별, 인기 순위를 고려하여 200개가 넘는 웹툰 중에 선정하였습니다. 사실 연재작 전부를 데이터셋으로 활용하고 싶었으나 컴퓨터 용량과 학습 시간을 고려하여 20개의 작품만 선정하였습니다. 

- 데이터 수집은 웹 크롤링을 통해 진행하였습니다. `config.json` 파일에 각 웹툰의 url 주소를 저장하였고, 해당 url 주소에 있는 이미지를 크롤링하였습니다. 이미지는 웹툰 당 500장을 모아 총 1만장의 이미지를 수집했습니다. 또한 시각화를 위해 각 웹툰의 썸네일도 수집하였습니다. 

### 데이터 전처리

<p align="center"><img src="https://ifh.cc/g/o7kQH7.jpg" width="700px"></p><p align="center">[그림 2] 수집한 이미지 1장의 형태 및 전처리 과정</p>

- 웹툰은 일반적인 만화와 달리 스마트폰에 최적화 되어 있어 수집된 이미지의 세로 길이가 길고 가로 길이는 짧습니다. 또한 이미지 크롤링을 진행하면 만화 컷별로 이미지가 수집되지 않고, 특정 이미지의 크기에 맞는 컷이 수집됩니다(저작권 문제가 있기 때문에 수집한 이미지를 직접 올리기는 힘들 것 같습니다). 위의 이미지와 같이 한 장의 이미지에는 여러 개의 컷과 말풍선, 공백이 포함되어 있습니다. 

- 저는 그림체를 분류하는 작업이기 때문에 이미지의 컷 부분만 잘라서 가져와야 된다고 생각했습니다. 처음에는 opencv를 활용하여 rectangle recognition을 통해 컷 부분만 crop하는 작업이 쉬울 것 같았습니다. 하지만 위의 그림에서 볼 수 있다시피 말풍선이 컷을 뚫고 지나가는 연출이 많아 rectangle recognition이 제대로 이뤄지지 않았습니다. 

- 저는 이러한 문제를 해결하기 위해 **이미지 내의 공백을 기준으로 컷을 분리**했습니다. 이미지의 가로 중심 좌표를 기준으로 세로로 내려오면서 RGB값을 픽셀 단위로 확인했습니다. 웹툰에서 공백 부분은 흰 색입니다. 확인한 RGB값이 (255, 255, 255)인 경우, 즉 흰 색(공백)인 경우 컷의 시작점으로 보았으며, 다시 RGB값이 등장하는 픽셀 좌표를 컷의 끝 점으로 보았습니다. 그 다음 컷의 시작점과 끝 점을 기준으로 crop하여 이미지 내에서 컷을 수 있었습니다. 

- 컷 내에서 RGB값이 (255, 255, 255)인 경우가 존재하기 때문에 threshold를 두고 threshold 이후의 픽셀값에 대해서만 끝점으로 인식하도록 코드를 짰습니다. 

### 모델 학습

#### Style Transfer

저는 그림체를 분류하기 위해 **Style Transfer** 방식을 활용하였습니다. 제가 활용한 방식을 설명하기에 앞서 먼저 Style Transfer부터 가볍게 살펴보도록 하겠습니다. 

<p align="center"><img src="https://bloglunit.files.wordpress.com/2017/04/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2017-05-16-e1848be185a9e18492e185ae-1-50-07.png?w=740
"></p><p align="center">[그림 3] Style Transfer</p>

- Style Transfer은 CNN 모델을 통해 하나의 이미지의 스타일을 다른 이미지에 덧입히는 기법입니다.  Style transfer은 pre-trained CNN 모델에서 스타일을 입히고 싶은 이미지(content image)와 스타일을 추출하고자 하는 이미지(style image)의 feature map을 각각 추출합니다. 이후 초기화된 랜덤 노이즈(output image)가 style image와 스타일이 비슷하도록, content 이미지와는 내용이 비슷해지도록 최적화합니다. 

- 이 때 사용되는 Loss function은 Content loss와 Style loss의 합으로 구성되어 있습니다. content loss는 output image와 Smooth L1 loss를 통해 구합니다. 반면 **style loss**는 **Gram matrix**를 구한 후 Gram matrix 간의 Frobenius norm의 제곱으로 정의합니다. 

- style loss가 이와 같이 복잡한 이유는 **style**은 각 feature map의 channel이 가지는 상관관계를 의미하기 때문입니다. 이를 위해서 각 layer별로, 각 activation별로 matrix를 구한 후 이를 비교해주는 연산이 필연적입니다. 자세한 내용은 [Andrew Ng](https://www.youtube.com/watch?v=QgkLfjfGul8&list=PLO3wjWsJ4jjzv8-6SsVyU2cPB8GgUIb5o&index=5) 교수님의 강의를 참고하시면 좋을 것 같습니다. 

#### Style Extraction

저는 이러한 Style Transfer에서 content loss를 구하는 부분을 제거하고 style loss만을 활용했습니다. 더 정확히 말씀드리면 이미지의 style만을 추출하는 작업을 수행하여 웹툰의 그림체를 추출했습니다. Style Transfer는 style image와 output image의 Gram Matrix를 각각 구한 후 비교하여 loss를 구했습니다. 하지만 저는 웹툰의 style에 대한 정보만이 필요하기 때문에 원본 이미지에 대한 Gram matrix를 구해주기만 하면 됩니다.  프로젝트에 사용된 주요 코드 살펴보도록 하겠습니다. 

```python
class Resnet(nn.Module):
	def __init__(self):
		super(Resnet,self).__init__()
		resnet = models.resnet50(pretrained=True)
		self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
		self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
		self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
		self.layer3 = nn.Sequential(*list(resnet.children())[5:6])

	def forward(self,x):
		out_0 = self.layer0(x)
		out_1 = self.layer1(out_0)
		out_2 = self.layer2(out_1)
		out_3 = self.layer3(out_2)

		return out_0, out_1, out_2, out_3
```

먼저 이미지의 feature map은 pre-trained된 ResNet을 통해 추출했습니다. 이미지의 content, 즉 객체나 조형에 대한 feature는 상대적으로 저차원적인 특징이기 때문에 얕은 layer에서 추출할 수 있는 특징입니다. 반면 이미지의 **style은 상대적으로 고차원적인 특징이기 때문에 깊은 layer**에서 추출할 수 있습니다. 이를 감안하여 ResNet이 style이라는 고차원적인 feature를 잘 추출할 수 있도록 layer의 깊이를 깊게 설정해주었습니다. 그리고 지정한 layer별로 feature map을 반환하도록 설계하였습니다. 

```python
def style_extract(data):

	total_arr = []
	label_arr = []
	for idx,(image,label) in enumerate(data):
		i = image.cuda()
		i = i.view(-1,i.size()[0],i.size()[1],i.size()[2])
		style_target = list(GramMatrix().cuda()(i) for i in resnet(i))
		arr = torch.cat([style_target[0].view(-1),style_target[1].view(-1),style_target[2].view(-1),
						style_target[3].view(-1)],0)
		gram = arr.cpu().data.numpy().reshape(1,-1)
		total_arr.append(gram.reshape(-1))
		label_arr.append(label)

		if idx % 50 == 0 and idx != 0:
			print(f'{idx} images style feature extracted..[{round(idx / len(data), 2) * 100}%]')
			print("\nImage style feature extraction done.\n")

	return total_arr, label_arr
```

 이후 각 layer로부터 추출한 feature map을 활용하여 Gram Matrix를 구한 후 이를 `total_arr` 변수에 저장하고 있습니다.  이를 통해 웹툰의 style과 label을 효과적으로 추출할 수 있었습니다. 위에서 보실 수 있다시피 단순히 Gram Matrix를 통해 feature에 대한 정보를 담고 reshape만 해준 채로 저장하는 것을 확인하실 수 있습니다. 

#### Style Classification
```python
def tsne(total_arr, n_components, perplexity, config='config.json'):
	config = configInfo(config)
	model = TSNE(n_components=2, init='pca',random_state=0, verbose=3, perplexity=100)
	result = model.fit_transform(total_arr)

	return result
```

마지막으로 t-SNE를 활용하여 웹툰별로 추출한 feature를 입력으로 받아 2차원 벡터를 얻습니다. t-SNE는 차원 축소 및 시각화에 널리 쓰이는 방법입니다. 고차원의 style에 대한 행렬을 2차원 벡터로 축소하여 시각화하는 것이 가능해집니다. sklearn에 내장된 t-SNE를 활용하였습니다. 

#### Style Visualization
<p align="center"><img src="https://user-images.githubusercontent.com/35513025/82119913-2dba0400-97bd-11ea-952e-e1034116f9fe.jpg
"></p><p align="center">[그림 4] Style Visualization by image</p>

<p align="center"><img src="https://user-images.githubusercontent.com/35513025/82119820-7fae5a00-97bc-11ea-9bdc-7871266662ac.jpg
"></p><p align="center">[그림 5] Style Visualization by webtoon</p>

[그림 4]는 사용한 이미지별로 그림체를 시각화한 결과입니다. 실제로 같은 웹툰일수록 같은 비슷한 좌표에 몰려있는 것을 확인할 수 있습니다. [그림 5]는 웹툰별로 각 컷별 좌표의 평균을 낸 후 썸네일을 해당 위치에 넣어 시각화한 결과입니다. 이를 통해 웹툰 간 유사도를 시각적으로 확인하는 것이 가능합니다. 예로 "갓!김치", "가타부타타", "가비지타임", "겟라이프", "갓물주"는 좌표상 서로 가까운 위치에 있어 어느 정도 비슷한 style을 공유한다는 것을 알 수 있습니다. 보다 많은 웹툰에 대한 데이터를 수집한 후 k-means clustering을 진행한다면 웹툰별로 유사한 그림체에 대한 정보를 얻는 것이 가능해질 것 같습니다. 

<br>

### 결론

지금까지 수행했던 프로젝트 중 가장 전형적이지 않은 프로젝트이지 않았나 싶습니다. 

- 지금까지는 주로 공개된 데이터셋을 사용했지만, 이번 프로젝트에서는 학습 데이터를 크롤링을 통해 처음부터 끝까지 직접 수집했습니다(사실 "웹툰"이라는 창작물이기에 당연하다고 생각합니다. 저도 데이터를 수집하면서 유출되지 않도록 각별히 주의했습니다).

- 데이터 전처리 또한 일반적인 crop, resize 정도가 아니라 웹툰이라는 이미지의 특성을 고려해야했기 때문에 까다로웠습니다. 최종적으로 적용한 데이터 전처리 방식을 떠올리기까지 여러 가지 방법을 시도해보았습니다. 
 
- CNN에 대한 근본적인 이해할 수 있는 좋은 기회였던 것 같습니다. 이전까지 이미지 분류, 객체 탐지 등과 같은 제한된 목적을 위해 CNN 모델을 사용하여, 모델이 마치 블랙박스처럼 느껴졌습니다. 하지만 Style Transfer를 적용하면서 feature map에 대해 심도있게 알게 되어 CNN 내부 동작 원리를 보다 더 잘 알게 된 것 같습니다. 

- 코드 작성면에서도 재사용이 가능하도록 모듈화에 신경을 썼습니다. pytorch template를 참고하여 프로젝트를 좀 더 체계적으로 구성할 수 있었던 것 같습니다. 

### 참고자료

[프로젝트 github 저장소](https://github.com/herbwood/webtoon_style_visualization)    
[style transfer에 대해 잘 설명한 Lunit Tech님의 블로그](https://blog.lunit.io/2017/04/27/style-transfer/)    
[Gram Matrix에 대해 진짜 잘 설명한 블로그](https://m.blog.naver.com/atelierjpro/221180412283)      
[Andrew Ng 교수님이 Style transfer에 대해 설명한 유튜브 영상](https://www.youtube.com/watch?v=QgkLfjfGul8&list=PLO3wjWsJ4jjzv8-6SsVyU2cPB8GgUIb5o&index=5)    
[t-SNE에 대해 잘 설명해준 ratgo님의 블로그](https://ratsgo.github.io/machine%20learning/2017/04/28/tSNE/)    
[Naver 웹툰 사이트](https://comic.naver.com/webtoon/weekday.nhn)      
[pytorch template github 저장소](https://github.com/victoresque/pytorch-template)  


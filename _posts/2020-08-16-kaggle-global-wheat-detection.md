---
layout: post
title:  "[Project] Kaggle Global Wheat Detection Competition"
subtitle:   "global wheat detection"
categories: study
tags: project
comments: true
use_math : true
---

이번 포스팅에서는 제가 참가한 [Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection) 참가 여정을 소개하도록 하겠습니다. 첫 캐글 도전인만큼 많은 것들을 배울 수 있었던 것 같습니다. 평소 Object detection 논문을 꾸준히 읽고 있었고, 배웠던 내용을 적용해보고자 도전하게 되었습니다. 

## 프로젝트 개요

[Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)  대회는 밀이 촬영된 이미지에서 **"밀알(wheat heads)"** 를 찾아내는 Object detection 대회입니다. 캐글 대회 도전은 처음이라 description이나 evaluation 방식 등을 제대로 읽어보지 않고 맨땅에 헤딩하듯 도전하여 약간의 시간이 소요되었던 것 같습니다. 처음부터 끝까지 제가 코드를 짜보고 싶었으나, 몇 번 시도 후 제가 최전선에서 적용하고 있는 다양한 기법들에 대해 잘 알지 못했으며, 밑바닥부터 시작하는 것은 지나치게 많은 시간이 걸린다고 생각하여 다른 분들인 작성한 notebook을 참고하여 도전했습니다.  저는 주로 [Alex Shonenkov](https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet)님의 notebook을 참고하였습니다. 

<br>

## 프로젝트 진행 과정

### 대회 설명

- [Global Wheat Head Dataset](http://www.global-wheat.com/2020-challenge/)을 데이터셋으로 사용합니다. 
- 평가 지표(evaluation metrics)는 mAP(mean Average Precision)입니다. 
- Kaggle Notebook를 제출하는 대회입니다
- 결과는 image_id에 대한 confidence score, bounding box의 x, y 좌표, width, height를 csv 파일 형식으로 제출하면 됩니다. 
- 대회에서는 자원에 대한 사용량을 지정했습니다. GPU 사용량 6시간 이하, 인터넷 접속 불가가 이에 해당합니다. 기타 제출 파일명은 `submission.csv`입니다. 제가 이 부분에서 인터넷 접속이 불가능하다는 정보를 간과해서 수많은 submission에서 error가 발생했습니다.... (Kaggle Notebook 디폴트값인 인터넷 접속 허가 상태로 제출을 했습니다;)

### Dependencies

제가 다른 분들의 notebook을 참고하면서 많은 딥러닝 관련 패키지를 알게 되었습니다.

- [timm](https://pypi.org/project/timm/) 은 최신 딥러닝 모델 및 optimizer, loss function을 제공해주는 pytorch 기반의 CNN 모델을 제공하는 패키지입니다. model에서는  ResNet과 같이 익숙한 모델은 물론, EfficientNet과 같은 최신 모델 역시 탑재했습니다. 
- [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)는 pytorch 기반 EfficientDet 모델을 구축한 github 저장소입니다. 내부에는 pre-trained된 EfficientDet과 가중치가 저장되어 있습니다. 이를 패키지 형식으로 다운받아 사용했습니다. 
- [albumentations](https://github.com/rwightman/efficientdet-pytorch)는 data augmentation 전용 라이브러리입니다. Flip, Rotation 등 기존 tensorflow, pytorch에서 제공하던 data augmentation이 모두 가능합니다. 무엇보다도 속도 측면에서 월등한 성능을 보여준다고 합니다. 또 다른 장점은 함수 내 p 파라미터를 통해 augmentation을 적용할 확률을 지정할 수 있습니다. 또한 pytorch와 호환성이 매우 뛰어납니다. 
- [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)는 WBF 방식을 통해 겹쳐있는 예측 bounding box를 효과적으로 제거해주는 패키지입니다. 자세한 내용은 아래에서 살펴보도록 하겠습니다. 

### 데이터 수집

데이터셋의 구조는 아래와 같습니다. train.csv는 image_id 및 bounding box의 좌상단, 우하단 x, y 좌표가 csv 파일 형식으로 저장되어 있습니다. 

```
global-wheat-detection/
|
|-- test // 테스트 이미지 데이터셋
|	 |--test_image01.jpg // test image
	  ...
|	 |--test_image10.jpg // test image
|-- train  // 학습용 이미지 데이터셋
|	 |--train_image01.jpg // train image
	  ...
|	 |--train_image01000.jpg // train image
|-- sample_submission.csv  // 제출 양식 
|-- train.csv  // 학습용 이미지 메타 데이터
|	
```
<p align="center"><img src="https://ifh.cc/g/vf0H3w.png" width="600px"></p><p align="center">[그림 1] 데이터셋 시각화</p>

train.csv에 저장된 image_id에 해당하는 이미지 파일을 train 디렉터리에서 찾고, bounding box의 x, y 좌표, width, height가 csv 파일 형식으로 저장되어 있습니다. ng box 좌표를 그려 시각화하는 것이 가능합니다. 학습용 이미지를 임의로 추출하고 bounding box를 그려 시각화한 결과는 위와 같습니다. sub title은 train.csv에서 읽어온 image_id입니다. 각 이미지의 크기는 1024 x 1024 입니다. 


### 데이터 전처리

이미 상당 수준 정제된 데이터였기에 전처리하는 과정은 크게 필요하지 않았습니다. 다만 Data Loader를 Custom했습니다. 

```python
class DatasetRetriever(Dataset):
    
    def __init__(self, image_ids, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        # index에 해당하는 이미지를 읽어들인 뒤 전처리
        image_id = self.image_ids[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        # data augmentation 적용
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape
```

pytorch를 사용하여 data load 시 데이터의 종류, 형식 등에 맞게 data loader를 custom해주는 과정이 필요하다는 것을 알게 되었습니다. pytorch의 Dataset 클래스를 상속받아 `DatasetRetriever` 클래스를 정의해줬습니다. 

- `__init__` : image_id, data augmentation 정의
- `__getitem__` : data loader를 슬라이싱, 인덱싱할 경우 반환할 값을 정의해줍니다. image_id의 인덱스에 맞는 이미지를 읽어들인 후 normalize해주는 과정을 거쳐 이미지 및 image_id를 반환합니다. 이 때 data augmentation을 적용할 수 있습니다. 
- `__len__` : data loader에 len 메서드를 적용할 경우 반환할 값을 지정합니다. 


### 모델 학습

#### 모델 정의

먼저 사용할 모델부터 살펴보도록 하겠습니다. 모델은 앞서 살펴본 timm 라이브러리에 내장된 EfficientDet 모델을 사용합니다. 이 때 EfficientDet 클래스에 config 인자를 전달하여 사용할 버전 및 가중치 값을 지정합니다. 

```python
tf_efficientdet_d5=dict(
        name='tf_efficientdet_d5',
        backbone_name='tf_efficientnet_b5',
        image_size=1280,
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5-ef44aea8.pth',
    ),
```

config 정보는 모델별 각종 설정 정보를 포함하고 있는 `efficientdet_model_param_dict`에 저장되어 있습니다. 위의 코드는 이번에 사용한 tf_efficientdet_d5에 대한 정보입니다. backbone 모델명, 입력 이미지 크기, feature pyramid channel, feature pyramid cell repeats, box classes, backbone argument, model weight download url에 대한 정보를 담고 있습니다. 

```python
def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5') # tf_efficientdet_d5 설정 저장
    net = EfficientDet(config, pretrained_backbone=False)  # EfficientNet 클래스에 설정 포함

    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # garbage collector
    del checkpoint
    gc.collect()
    
    # evaluation mode
    net = DetBenchEval(net, config)
    net.eval();
    return net.cuda()

net = load_net('../input/wheat-effdet5-fold0-best-checkpoint/fold0-best-all-states.bin')
```

위의 config 설정을 EfficientDet에 전달한 후, bounding box의 좌표와 confidence score를 예측할 HeadNet을 class_net으로 지정하여 최종 모델을 정의합니다. 

#### TTA(Test Time Augmentation)

이번 캐글 대회를 통해 새로 알게 된 기법이 있습니다. 바로 TTA(Test Time Augmentation)입니다. TTA는 추론 시 augmentation이 적용된 각각의 이미지에 대해서 결과를 예측하고, 이 값들의 평균을 예측값으로 사용하는 방법입니다. 이를 위해 사전에 적용할 augmentation 방법을 클래스로 정의했습니다.

```python
# 좌우 반전 augmentation
class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    
    # 이미지 좌우 반전 후 x, y의 좌표를 원래대로 돌려주는 함수
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes
```

위의 코드는 추론 시 적용할 좌우 반전 augmentation을 정의한 예시입니다. 이와 같은 방식으로 상하 반전, 90도 회전 augmentation 클래스 역시 정의했습니다(BaseWheatTTA 클래스는 추상 클래스입니다). augmentation을 적용하면 bounding box의 좌표 역시 바뀐다는 문제가 있습니다. 이를 해결하기 위해 deaugment_boxes 메서드를 통해 bounding box의 좌표 역시 바꿔줍니다. 

<p align="center"><img src="https://ifh.cc/g/AHi4K0.jpg" width="600px	"></p><p align="center">[그림 2] TTA 결과 시각화</p>

위의 그림은 원본 이미지와 90도 회전 augmentation을 적용한 이미지, 그리고 augmentation이 적용된 이미지를 원래대로 되돌리는 이미지를 보여줍니다. 앞서 언급한 deaugment 메서드가 잘 동작한 것을 확인할 수 있습니다. 

```python
def make_tta_predictions(images, score_threshold=0.25):
    with torch.no_grad():
        images = torch.stack(images).float().cuda()
        predictions = []
        for tta_transform in tta_transforms:
            result = []
            
            # TTA한 batch 데이터와 이미지 label을 1로 초기화한 값을 이미지 수만큼
            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                
                # 예측한 bounding box의 값이 score_threshold 이상일 경우
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                
                # TTA하기 이전 box형태로 되돌려줌
                boxes = tta_transform.deaugment_boxes(boxes.copy())
                
                # box의 좌표와 confidence score 저장
                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })
                
            # 여러 combination의 TTA 조합 중 하나의 TTA를 실험한 결과의 prediction 저장
            predictions.append(result)
        
    return predictions
```

augmentation을 보다 다양하게 하기 위해서 augmentation 클래스를 다양하게 조합하는 방식을 사용합니다. `tta_transforms` 변수에는 제가 정의한 상하 반전, 좌우 반전, 90도 회전이라는 augmentation의 모든 조합이 저장되어  있습니다. 

- 이후 앞서 정의한 모델을 통해 augmentation이 적용된 이미지에 대한 결과를 예측합니다. 이 때 예측 결과는 threshold보다 iou가 높은 bounding box에 대한 좌표, confidence score입니다. 

- 이후 augmentation 조합별로 추론 결과를 `predictions` 변수에 저장합니다. 

#### WBF(Weighted Boxes Fusion)

`predictions`에 저장된 bounding box의 좌표와 confidence score는 불필요하게 많은 정보를 가지고 있을 수 있습니다. bounding box가 지나치게 많이 겹치는 경우 오히려 예측 성능이 떨어질 수 있기 때문에 Non-Max Suppression 등의 방법을 사용해왔습니다. 하지만 이번에 캐글을 도전하면서 [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)라는 새로운 기법을 알게 되었습니다. NMS는 겹친 예측 bounding box를 confidence score와 IOU를 통해 제거하여 가장 합리적인 bounding box만 남깁니다. 하지만 이와 달리 WBF는 예측된 모든 bounding box를 사용하여 예측 성능이 상당히 높아진다고 합니다. 

```python
# Weighted Box Fusion을 활용하여 최종 box, score, label 반환

def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels
```

위와 같이 `predictions`에 저장된 bounding box의 좌표와 confidence score를 파라미터에 지정된 값에 따라 전처리를 해준 후 WBF를 통해 최종적인 bounding box 좌표, confidence score, label(밀알 여부)를 반환합니다. 

### 추론

```python
results = []

for images, image_ids in data_loader:
    predictions = make_tta_predictions(images)
    for i, image in enumerate(images):
        boxes, scores, labels = run_wbf(predictions, image_index=i)
        boxes = (boxes*2).round().astype(np.int32).clip(min=0, max=1023)
        image_id = image_ids[i]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)
```

마지막으로 앞서 살펴본 내용들을 순차적으로 모두 적용하여 최종 결과를 예측합니다. `format_prediction_string` 메서드는 submission 형식에 맞게 예측 결과를 저장해주는 메서드이로 사전에 정의했습니다. 예측 결과는 submission.csv 파일로 저장했습니다. 

### 최종 결과
<p align="center"><img src="https://camo.githubusercontent.com/c9bcd14fd6bc37ec04bf15f7d57c35e0fe360566/68747470733a2f2f6966682e63632f672f744f765636572e6a7067" width="400px	"></p><p align="center">[그림 3] 추론 결과 시각화</p>

- 제출은 지금까지 설명한 코드와 Faster R-CNN에 Pseudo Labelling을 추가한 코드, 2개를 제출했습니다. Public에서는 Faster-RCNN 코드가 순위가 더 높았는데 Private 순위에서 뒤집어졌습니다.
- Private score에서 2245팀 중 최종 503등을 기록했습니다. 

<br> 

## 결론

이번에 캐글에 처음 도전하면서 정말 많은 것들을 알게된 것 같습니다. 

- 먼저 제가 처음으로 참여한 Object detection 대회입니다. 그 동안 꾸준히 Object detection 관련 논문들을 읽어오면서 현재까지의 발전사를 알게 되었지만 배운 내용을 직접 실습할 기회는 없었던 것 같습니다. 그 동안 알고 있었다고 믿었던 내용들을 다시 살펴보았고 잘못 알고 있었던 내용을 다시 살펴보는 좋은 기회였던 것 같습니다. 그리고 현업에서의 Object detection이 어떤 방식으로 진행되는지 전체 프로세스를 알 수 있었습니다. 

- 그리고 캐글 플랫폼에 대해 익숙해진 것 같습니다.  캐글 노트북 사용법, 대회 참가 시 유의사항, 유용한 노트북들 등, 대회에 참가하지 않더라도 캐글 자체가 정말 유용하다는 것을 알게된 것 같습니다. 

- 딥러닝 관련 새로운 개념들에 대해서도 많은 것을 알게 되었습니다. state-of-the-art 딥러닝 모델이 저장되어 있고, 업데이트가 굉장히 빠른 timm 패키지, 빠른 고성능 data augmentation 패키지 albumentations, 그리고 TTA, WBF 등 Object detection의 성능을 끌어올리기 위한 다양한 기법들을 배웠습니다. 

최종 순위 자체는 높지 않지만 참여함으로써 정말 많은 것을 배울 수 있었습니다. 앞으로는 이론적인 부분에 대한 공부는 물론 캐글에서 컴퓨터 비전 대회가 열리면 적극적으로 참가해볼 생각입니다. 

## 참고자료
[Kaggle Global Wheat Detection 대회](https://www.kaggle.com/c/global-wheat-detection)    
[프로젝트 저장소](https://github.com/herbwood/kaggle_global_wheat_detection)    
[많은 부분 참고한 Alex Shonenkov님의 notebook](https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet)    
[efficientdet-pytorch 패키지](https://github.com/rwightman/efficientdet-pytorch)     
[timm 패키지](https://pypi.org/project/timm/)     
[albumentations 라이브러리](https://github.com/rwightman/efficientdet-pytorch)      
[TTA에 대해 잘 설명한 블로그](https://hwiyong.tistory.com/215)    
[Weighted Boxes Fusion 저장소](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)       
[WBF에 대해 잘 설명한 블로그](https://lv99.tistory.com/74)   

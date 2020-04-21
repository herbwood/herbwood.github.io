---
layout: post
title:  "[Project] Pytorch 프로젝트 Template"
subtitle:   "pytorch template"
categories: study
tags: project
comments: true
use_math : true
---

&nbsp;&nbsp;&nbsp;&nbsp;딥러닝 관련 github repository를 살펴보면 파일과 디렉터리가 특정 형식에 맞게 정리된 모습을 확인할 수 있습니다. 처음에는 이러한 구조를 이해하는 데 어려움이 있었지만 이후에 딥러닝 프로젝트들이 **template**에 맞게 저장되어 있음을 알게 되었습니다. 딥러닝 프로젝트 github 저장소를 보다 잘 이해하고, 직접 프로젝트를 구성할 때 참고하기 위해 딥러닝 프로젝트 template을 정리해보았습니다. Template 분석을 위해 [pytorch-template](https://github.com/victoresque/pytorch-template) github 저장소를 참고했습니다. 

![Cap 2020-03-23 10-42-32-035](https://user-images.githubusercontent.com/35513025/77271961-5fe64280-6cf3-11ea-91ca-3c2177c2095d.jpg)

프로젝트 전체 구조는 다음과 같습니다.   

```bash
pytorch-template/
    │
    ├── train.py - main script to start training
    ├── test.py - evaluation of trained model
    │
    ├── config.json - holds configuration for training
    ├── parse_config.py - class to handle config file and cli options
    │
    ├── new_project.py - initialize new project with template files
    │
    ├── base/ - abstract base classes
    │   ├── base_data_loader.py
    │   ├── base_model.py
    │   └── base_trainer.py
    │
    ├── data_loader/ - anything about data loading goes here
    │   └── data_loaders.py
    │
    ├── data/ - default directory for storing input data
    │
    ├── model/ - models, losses, and metrics
    │   ├── model.py
    │   ├── metric.py
    │   └── loss.py
    │
    ├── saved/
    │   ├── models/ - trained models are saved here
    │   └── log/ - default logdir for tensorboard and logging output
    │
    ├── trainer/ - trainers
    │   └── trainer.py
    │
    ├── logger/ - module for tensorboard visualization and logging
    │   ├── visualization.py
    │   ├── logger.py
    │   └── logger_config.json
    │  
    └── utils/ - small utility functions
        ├── util.py
        └── ...
```

## base
data_loader, model, trainer의 **추상 클래스** 모듈입니다. 추상 클래스를 생성하지 않고 각각의 파일에 직접 클래스를 작성하는 경우도 있었습니다. 하지만 본 template는 하나의 파일에 클래스를 작성 시 코드가 지나치게 길어져 가독성이 떨어지는 것을 방지하고자 별도의 추상 클래스 모듈을 생성해준 것 같습니다.   


**data_loader.py**
- 디렉터리로부터 데이터셋을 불러오는 기능을 제공합니다. 
- train, validation 데이터셋을 생성자를 통해 입력받은 비율에 따라 분리하는 기능도 제공합니다.   

**model.py** 
- 모델의 forward 메서드를 abstractmethod로 지정하여 상속받는 클래스에 세부 사항을 필수적으로 기재하도록 지정하였습니다.
-  문자열 메서드를 통해 학습 가능한 파라미터의 수를 출력하도록 하였습니다.  

**trainer.py`**  
- 생성자로 logger, 사용할 gpu 수, loss function, metric, optimizer, epoch, save_period, early stopping 여부, tensorboard 사용 여부 등을 지정했습니다.
- epoch별로 실행할 train 메서드는 추상화메서드(@abstractmethod)로 지정했습니다.
- 전체 train과정에서 epoch별로 loss, accuracy 등을 출력하도록 하고, checkpoint에 저장했습니다. 또한 학습이 중단되는 것에 대한 대책으로 마지막 checkpoint로부터 필요한 값을 불러들이는 resum checkpoint 메서드를 제공합니다. 또한 가장 높은 정확도를 보인 epoch의 값들을 저장합니다.   

(epoch 한 차례가 끝나면 다음과 같은 값을 출력됩니다)  

![Cap 2020-03-20 17-42-27-620](https://user-images.githubusercontent.com/35513025/77149062-22579e80-6ad4-11ea-8473-7d59ea40cb28.png)


## data_loader  
base_data_loder를 상속받아 실제 데이터를 load하는 모듈입니다.  

**data_loader.py** 
데이터가 저장된 디렉터리, batch size 등을 지정하고 torchvision 메서드인 transform을 통해 data augmentation을 진행할 수 있습니다.   


## logger  
logging과 tensorboard 시각화를 위한 기능을 제공하는 모듈입니다.  

**logger_config.json**
logger에 대한 정보가 저장된 json 파일입니다.   

**logger.py**
logger_config.json에 저장된 logger에 대한 정보를 읽어들여 load시킵니다.   

**visualization.py**
tensorboard를 통해 train 결과를 시각화하는 기능을 제공합니다. 

## model  
train, evaluation 시 필요한 loss function, metric, model을 지정하는 모듈입니다.  

**loss.py**
torch.nn.functional에서 제공하는 메서드를 사용하거나 custom한 loss function을 지정할 수도 있습니다. 

**metric.py** 
모델 성능 평가 시 사용할 metric을 지정합니다. 본 template에서는 accuracy와 top_k_acc를 metric으로 설정했습니다. 

**model.py**
base_model를 상속받아 모델을 설계합니다. 

## trainer
base_trainer를 상속받아 epoch별로 train할 세부 사항을 지정합니다.  

**trainer.py** 
- 생성자로 학습할 데이터셋, model, loss function, metrics, optimizer 등을 지정합니다
- epoch별로 학습할 방법을 지정하고 학습 현황을 logger를 통해 출력하고 log 파일을 업데이트 합니다
- validation 시 epoch별로 수행할 메서드를 지정합니다

(학습 시 logger를 통해 다음과 같이 학습 현황이 출력됩니다)
![Cap 2020-03-20 22-50-55-799](https://user-images.githubusercontent.com/35513025/77169904-9ce8e400-6afd-11ea-894e-9caef5003bc7.png)


## utils
학습 시 필요한 기타 기능을 모아둔 모듈입니다

**util.py**
- json 파일을 읽고 쓰는 기능을 제공합니다
- data loader를 반복자 형태로 반환하는 기능을 제공합니다
- MetricTracker 클래스를 통해 pandas 데이터프레임에 현재까지의 loss를 기록합니다

- data loader에 대한 정보(디렉터리명, batch size, validation split 등), optimizer(learning rate, weight decay 등), loss, metric 등 하이퍼  파라미터에 대한 정보를 가지고 있는 json 파일

## parse_config.py

하이퍼파라미터가 저장돈 config.json 파일을 파싱하고 CLI(Command Line Interface)옵션을 처리하기 위한 클래스입니다.  

- CLI를 통해 입력받은 값에 따라 config 정보를 업데이트 시킵니다. 이를 구현하기 위해 @classmethod를 활용합니다.
- logger에 대한 옵션을 초기화시킵니다.

*생소한 모듈들이 많이 등장해 분석하기가 어려웠습니다. 프로젝트를 직접 구성하면서 좀 더 깊게 파볼 계획입니다...*

## train.py

- 모델에 대한 학습을 진행하는 파일입니다.

- CLI를 통해 입력받은 옵션을 config파일에 반영하고 config 파일로부터 하이퍼파라미터에 대한 모든 정보를 입력받아 학습을 시작합니다.

## test.py

- 학습된 모델에 대한 평가를 수행하는 파일입니다. 

- CLI를 통해 입력받은 옵션을 config파일에 반영하고 config 파일로부터 하이퍼파라미터에 대한 모든 정보를 입력받아 평가를 진행합니다.


## Conclusion
&nbsp;&nbsp;&nbsp;&nbsp;**Pytorch Template**을 분석하면서 단번에 이해하기는 힘들었지만 프로젝트 구조가 전반적으로 어떻게 구성되있는지 파악할 수 있었습니다. 앞으로는 딥러닝 프로젝트 github 저장소를 보더라도 너무 당황하지 않고 차분하게 코드를 살펴볼 수 있을 것 같습니다. Template을 좀 더 깊게 이해하기 위해 생소한 모듈이나 객체 지향 프로그래밍 개념에 대해 공부해볼 계획입니다. 그리고 아직 로컬에 GPU가 없지만(ㅠ) 프로젝트를 template에 맞게 구성하는 습관도 들이고자 합니다. 

## Reference
[Pytorch Template](https://github.com/victoresque/pytorch-template)  
[logging에 대해 상세히 설명한 블로그](https://hwangheek.github.io/2019/python-logging/)  
[Python argparse 라이브러리 공식 문서](https://docs.python.org/ko/3.7/howto/argparse.html#id1)  
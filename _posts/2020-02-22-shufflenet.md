---
layout: post
title:  "[DL] ShuffleNet v1 논문 리뷰"
subtitle:   "ShuffleNet v1"
categories: study
tags: dl
comments: true
use_math : true
---

최근 핸드폰, 자율주행차 등 하드웨어에서 딥러닝을 사용하기 위한 모델의 경량화 연구가 활발히 진행되고 있습니다. 모델 경량화를 위해 모델 구조 변경, Weight Pruning, Quantization 등과 같은 방법이 사용되고 있습니다. 이번 포스팅에서는 지난 MobileNet와 유사하게 효율적인 Convolution filter를 설계하여 경량화시킨 모델인 [ShuffleNet](https://arxiv.org/pdf/1707.01083.pdf)에 대해 살펴보도록 하겠습니다. 

## Prior Work

- 논문의 저자는 기존 모델이 엄청난 complexity를 부여하는 1x1 convolution(pointwise conv)을 충분히 고려하지 않았다고 봅니다. 가령 ResNext는 computational cost를 조정하기 위해 3x3 layer에만 group conv를 적용하였고, 그 결과 각 residual unit에서 pointwise conv(1x1 conv)는 93.4%라는 엄청난 연산량을 차지하게 됩니다. 

- 이러한 문제를 해결하기 위해서는 1x1 conv에도 group convolution을 적용하면 됩니다. group conv는 computational cost를 극적으로 줄여주기 때문입니다. 하지만 여러 개의 grop conv가 쌓여있다면 한 가지 부작용이 있습니다. 

- 바로 오직 특정 feature map만이 특정 group의 input channel로부터 비롯된다는 것입니다.  즉 group 간의 교류(cross talk)가 없어 각 gruop 사이에서만 정보가 흐른다는 것입니다. 이러한 성질은 feature map간의 정보의 흐름을 막고 모델의 표현력을 약하게 만듭니다. 


## Channel Shuffle for Group Convolution

![Cap 2020-05-11 17-54-43-437](https://user-images.githubusercontent.com/35513025/82120988-c7d17a80-97c4-11ea-83b2-c1b871b29b89.png)
<p align="center">[그림1] Channel Shuffle for Group Convolution</p>

- 만약 group conv가 다른 그룹의 input channel을 얻을 수 있게 된다면 입력값과 출력값 channel은 서로 완전한 관계를 맺게 될 것입니다. 논문의 저자는 이를 구현하기 위해 **Channel Shuffle**이라는 방식을 제안합니다. 각 group들에 있는 channel들을 shuffle해 버려서 input 과 output의 channel 들이 fully related, 즉 모두 관계를 가지도록 만드는 방법입니다.

- Channel Shuffle은 그림 1과 같은 과정을 거칩니다. 먼저 각 그룹의 채널을 서브그룹으로 나눈뒤 channel shuffle을 통해서 group별 채널을 섞어줍니다. 그리고 그룹 내의 채널을 작은 서브그룹으로 나눠준 후 다음 층의 그룹에 서로 다른 서브구룹을 입력값으로 사용합니다. 이는 구현 코드를 살펴보면 더 자세히 살펴볼 수 있습니다. 

```python
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
```
- 위의 코드는 [torchvison zoo](https://pytorch.org/docs/stable/_modules/torchvision/models/shufflenetv2.html#shufflenet_v2_x0_5)에 공개된 Channel Shuffle을 pytorch 패키지를 통해 구현한 코드입니다. channel_per_group에 group별 channel 수를 구하고 transpose를 통해 shuffle해준 후 flatten하여 각 group에 할당한 것을 확인할 수 있습니다. 


## ShuffleNet Unit
다음으로 네트워크 전반에 사용되는 ShuffleNet Unit에 대해 살펴보도록 하겠습니다.

![Cap 2020-05-11 17-54-56-788](https://user-images.githubusercontent.com/35513025/82120989-ca33d480-97c4-11ea-9818-6ba5aee5d0e4.png)
<p align="center">[그림 2] ShuffleNet Unit</p>

- (a)는 일반적인 Resnet의 bottleneck입니다. 

- (b)는 pointwise conv - channel shuffle - depthwise conv - popintwise conv를 순차적으로 적용한 shufflenet unit입니다. (b)에서 처음에 pointwise conv를 적용한 이유는 왼쪽 shortcut path의 channel 수를 맞춰주기 위함이라고 합니다.  

- (c)는 실험에 최종적으로 사용될 unit으로, stride=2로 설정한 shufflenet unit입니다. (b) unit에 두 가지 변주를 주었습니다. 먼저 3x3 average pooling을 shortcut path에 추가했으며, 마지막단의 element-wise addition을 channel concat으로 대체했습다. Concat은 computational cost는 상대적으로 낮지만, channel 수를 효과적으로 늘릴 수 있습니다. 

- 이같은 ShuffleNet Unit으로 Shufflenet은 더 넓은 feature map을 사용할 수 있게 됩니다. 이는 작은 네트워크에서 매우 중요한데, 작은 네트워크는 Channel 수의 부족으로 성능이 상대적으로 낮기 때문입니다. 


## Network Architecture
![Cap 2020-05-11 17-55-05-491](https://user-images.githubusercontent.com/35513025/82120991-cbfd9800-97c4-11ea-8934-44b4f204962e.png)
<p align="center">[그림 3] ShuffleNet Network Architecture </p>

전체 네트워크 구조는 위와 같습니다. output channel의 수를 조정하면서 전반적인 computational cost가 140 FLOPS에서 변하지 않도록 유지했습니다. 더 많은 group의 수는 더 많은 output channle의 수를 의미하고 이는 곧 네트워크가 더 많은 정보를 가질 수 있음을 의미합니다. 

## Experiments

다음으로 논문에서 진행한 다양한 실험 결과에 대해 살펴보도록 하겠습니다. 

- group의 수가 더 많을 때가 그렇지 않을 때보다 더 좋은 성능을 보였습니다. group conv는 제한된 copmplexity에서 더 많은 feature map channel을 가능케하고 이는 곧 네트워크가 더 많은 정보를 학습할 수 있음을 의미합니다. 

- Chaneel Shuffle이 있었을 때가 없을 때보다 성능이 월등히 좋았습니다. 특히 group이 많을 때 그 효과가 극명했습니다. 

- MobileNet과 비교했을 때, 비슷한 complexity 하에서 성능이 월등히 좋았으며 실제 모바일 기기에서 inference 속도를 실험한 결과 AlexNet과 비교했을 때 비슷한 성능을 보였지만 속도는 13배 이상 빨랐습니다.

## Conclusion

ShuffleNet은 지난 포스팅에서 살펴보았던 MobileNet과 마찬가지로 Convolution 방식에 변화를 주어 모델 경량화를 이뤄냈습니다. 사실 Channel Shuffle이 구체적으로 어떻게 이뤄지는지 이해하기 힘들었는데 다행히 pytorch로 구현된 코드를 보고 이해할 수 있었습니다. 

## Reference
[ShuffleNet 논문](https://arxiv.org/pdf/1707.01083.pdf)  
[ShuffleNet 논문에 대해 설명한 블로그](https://gamer691.blogspot.com/2019/03/paper-review-shufflenet-extremely.html)  
[Different types of Convolutions](https://hichoe95.tistory.com/48)  
[ShuffleNet 구현 코드(pytorch)](https://pytorch.org/docs/stable/_modules/torchvision/models/shufflenetv2.html#shufflenet_v2_x0_5)  
[모델 경량화 연구 동향](https://ettrends.etri.re.kr/ettrends/176/0905176005/34-2_40-50.pdf)  



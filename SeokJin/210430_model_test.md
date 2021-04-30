모든 조건은 동일하게 하고 모델만 바꾸면서 실험하였다.

Model : segmentationmodels.pytorch efficientnet-b0 UNet

epoch : 8

batch_size : 8

image_size : 256

## 1. (mIoU : 0.5440)

Encoder: resnext50_32x4d

Decoder: DeepLabV3Plus

Encoder_weights: imagenet

    Validation #1 mIoU: 0.2979
    Validation #2 mIoU: 0.3477
    Validation #3 mIoU: 0.3955
    Validation #4 mIoU: 0.4145
    Validation #5 mIoU: 0.4246
    Validation #6 mIoU: 0.4275
    Validation #7 mIoU: 0.4170
    Validation #8 mIoU: 0.4328

## 2.

Encoder: resnext50_32x4d

Decoder: DeepLabV3Plus

Encoder_weights: ssl

    Validation #1 mIoU: 0.2697
    Validation #2 mIoU: 0.3081
    Validation #3 mIoU: 0.3546
    Validation #4 mIoU: 0.4001
    Validation #5 mIoU: 0.3836
    Validation #6 mIoU: 0.4152
    Validation #7 mIoU: 0.4229
    Validation #8 mIoU: 0.4114

## 3.

Encoder: resnext50_32x4d

Decoder: DeepLabV3Plus

Encoder_weights: swsl

    Validation #1 mIoU: 0.2818
    Validation #2 mIoU: 0.3355
    Validation #3 mIoU: 0.3805
    Validation #4 mIoU: 0.4037
    Validation #5 mIoU: 0.4152
    Validation #6 mIoU: 0.4036
    Validation #7 mIoU: 0.4014
    Validation #8 mIoU: 0.4244

## 4.

Encoder: efficientnet-b0

Decoder: DeepLabV3Plus

Encoder_weights: imagenet

    Validation #1 mIoU: 0.0064
    Validation #2 mIoU: 0.0063
    Validation #3 mIoU: 0.0064
    Validation #4 mIoU: 0.0062
    
## 5.

Encoder: resnext50-32x4d

Decoder: DeepLabV3Plus

Encoder_weights: imagenet

    [0.2934440396491227,
     0.3448902234220234,
     0.39674570463829517,
     0.4119239128420562,
     0.41075280312840273,
     0.43460735798174616,
     0.41354198273128473,
     0.4381222749826386]

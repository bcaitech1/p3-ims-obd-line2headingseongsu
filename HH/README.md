# 정리

## 210427

### 시도했던 내용

- 공통 setting 
  - learning_rate = 1e-4 
  - batch = 8 
  - random seed = 21 
  - weight_decay = 1e-6 
  - optimizer = Adam 
  - loss = cross entropy
  - seed = 21

1. 26일 daily_mission 에서 fcn16s를 pretrain된 VGG16을 이용하여 구현하였고 학습을 돌렸다. 
   너무느려서 epoch은 6만 돌려주었다.
   결과 : 0.3218
2. 27일 daily_mission 에서 sgenet을 완성하여 학습을 돌려보았다.
   epoch : 14
   결과 : 0.3075
3. torchvision에서 제공해주는 segmantic segmentaion 모델 DeepLabV3_resnet50 과 resnet101을 시도해보았다.
### 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

- pretrain된 모델을 사용하는 것이 성능이 더 좋은 것 같다.
- segNet 도 생각보다 너무 느리다...
- Augmentation을 시도해 보아야 겠다. (albumentation 라이브러리 사용)
- 3번 내용의 경우 CUDA 메모리 부족 현상이 발생... 좀 더 고민해 봐야 할 것 같다.



## 210428
### 시도했던 내용

- 공통 setting 
  -  learning_rate = 1e-4 
  - random seed = 21 
  - weight_decay = 1e-6 
  - optimizer = Adam 
  - loss = cross entropy 
  - epoch = 20
  - seed = 21

1. model : Baseline_DeepLabV3 ( 직접 구현 )
   + batch=8 
   + 결과 : 0.3673
2. model : Baseline_DeepLabV3 (직접 구현) / 
   + Augmentations 적용 (Resize(256,256), Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   + 결과 : 0.2749 
3. model : Baseline_fcn16s

   -  Augmentations 적용 (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion) 
   - back bone :  vgg16
   - 결과 : 0.4047

4. model : DeeplabV3

   - back bone : resnet50

   - Augmentation으로 normalize 만 적용
   - 결과 :  0.2124

5. model : DeeplabV3Plus

   - back bone : resnet50 
   - Augmentation : normalize   
   - 결과 : 0.1617

### 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

- 2번 실험을 통해서 resize는 속도는 줄여주지만 성능은 떨어진다 고로 성능향상에 도움이 되는 Augmentation을 찾을 때 와 같이 빠른 실험을 위해서 사용하면 좋을 것 같다.

- 3번 실험 결과를 보면 pretrain 모델을 사용한 fcn-16s가 직접 구현한 DeepLabV3 보다 성능이 더 잘 나오는 것을 확인할 수 있다.
- 어제 torchvision에서 제공하는 pretrain된 segmantic segmentation 모델을 사용할 떄 메모리 부족 현상이 일어났는데
   다른 모델을 학습시키고 kernel을 종료하지 않아서 계속 memory를 점유하고 있었다. 해결하고 현재 시도중.
- 5번 실험과 6번 실험의 결과가 너무 낮게 나왔다
   + 원인으로 추정되는 것은 dataset의 전처리 과정에서 픽셀 값의 정규화를 위해 255로 나누어준 부분을 제거하지 않고 normalize를 시도하여 학습이 원활하게 되지 않았던 것으로 생각한다. 수정 후 다시 학습을 시도해야 겠다. 



## 210429

### 시도했던 내용

- 공통 setting : 
  - lr = 0.0001 
  -  weight_decay = 1e-6
  - optimizer = Adam
  - loss = cross entropy
  - normalize 적용
  - seed = 21
  - Encoder weight : imagenet

1. model : DeeplabV3
   - backbone : resnet101 
   - Resize(256,256)
   - batch=24 
   - 결과 : 0.5070 

2. model : DeeplabV3Plus
   - backbone : resnet101
   - Augmentation : Resize(256,256)  
   - batch=24 
   - 0.5522

3. model : DeeplabV3Plus
   - resnet101
   -  Augmentation : Resize(256,256), Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion
   - batch=24 
   - seed=42 
   - 결과 : 0.5553

4. model : DeeplabV3Plus

   - backbone : resnet101

   - Augmentation : (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   - batch=4
   - 결과 :  0.5027
   - 학습 Validation mIoU: 0.4954

5. model : DeeplabV3Plus

   - backbone : resnet50 

   - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   - batch=16
   - 결과 : 0.5817

6. model : DeeplabV3Plus
   - backbone : resnext50
   - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   - batch=16 
   - 결과 : 0.5881
   - Validation mIoU: 0.5546

7. model : U-net 
   - backbone : EfficeintNet b4
   - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   - batch=8 
   - encoder_weight : noisy_study
   - 결과 :  0.5022 
8. model :  U-net 
   - backbone : EfficientNet b0
   - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
   - batch=16 
   - encoder_weight : noisy_study 
   - epoch=40 / 0.4757

### 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

- 같은 조건으로 실험한 1번과 2번 실험을 통해 DeepLabV3 보다는 DeepLabV3Plus가 성능이 더 좋다는 것을 확인했다
  - DeepLabV3Plus를 기반으로 encoder만 변경해서 더 좋은 성능을 가진 모델을 찾아보기
- 3번에서 실제 학습한 Validation의 mIoU와 리더보드의 mIoU가 크게 차이나지 않는 것을 볼 수 있다.
  - 기존에는 batch size 단위로 계산을 하여 실제값과 괴리감이 컸는데, 전체 validation set으로 계산을 하면 근사한 값으로 형성된다.
  - [aistages](http://boostcamp.stages.ai/competitions/28/discussion/post/256) 강민용 캠퍼님이 올려주신 코드로 수정하였다.( - - ) ( _ _ )
- 3번과 4번 실험을 통해서 Resize를 적용한 것과 적용하지 않은 것에서 성능차이를 보인다.
  - 512 * 512 로 학습을 하면 성능이 더 좋아질 것이라는 가정하에 실험을 진행했다. 그러나 생각과는 반대의 결과가 나왔다 아마 batch size가 너무 작아서 결과에 영향을 미치지 않았을까?
- 위의 실험 결과를 다시 검증하기 위해서 resnet50으로 다시 실험을 진행하였다.
  - resnet101로 실험 했던 결과들 보다 더 좋은 성능을 보여줬다.
  - Resize를 적용하게 되면 성능이 저하 되는 것을 확인 (3번과 5번 비교)
  - batch size가 너무 작은 것도 성능의 저하에 원인이 되는 것을 확인 (3, 4번을 비교 하고 5번의 결과를 다시보면 알 수 있다.)
- 위의 결과를 통해서 resnet50의 계열이 현재 나의 환경상태에서 최고의 Performance를 보여주고 있다. 그래서 이번엔 resnext50을 back bone 모델로 사용하여 결과를 보았다.
  - 6번 실험의 결과를 보면 지금까지 가장 좋은 성능을 보여주고 있다.
  - resnet101 < resnet50 < resnext50 이므로 앞으로의 실험에서는 resnext50또는 resnet50을 이용하도록 하자.
  - Resize(256, 256) < no Resize
  - batch size에 대해서는 정확한 답은 없는 것으로 알고 있다. 하지만 나의 실험을 통해서 더 좋은 모델은 resnet101은 결국  batch size가 너무 작아서 성능이 제대로 나오지 않은 것으로 생각된다. 적절한 batch size를 줄 수 있는 resnet50 계열의 모델을 사용하도록 해야겠다.
- Efficient + U-Net 모델도 성능이 나쁘지는 않지만 DeepLabV3Plus가 더 높은 성능을 보인다.
  - EfficientNet은 좋은 성능을 가진 높은 version을 이용하려고 할 수록  batch size를 많이 낮춰줘야 한다. 아마 이것이 원이이지 않을까 하는 생각을 한다.



## 210430

### 시도했던 내용

- 공통 setting : 
  - lr = 0.0001 
  - weight_decay = 1e-6
  - optimizer = Adam
  - loss = cross entropy
  - normalize 적용
  - seed = 21
  - Encoder weight : imagenet
  - model : DeepLabV3Plus
  - 기본 Augmentation : HorizonFliip, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion 을 사용합니다

1. 첫번째 - encoder 변경

   - encoder : se_resnext50_32x4d
   - encoder_weights : imagenet

   - Augmentation : Resize(256, 256), HorizonFliip, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion
   - batch_size : 16
   - 결과 : 0.5766
2. 두번째 - 새로운 Augmentation 적용

   - encoder : resnext50
   - batch_size : 16
   - 새로운 Augmentation 추가 : Transpose, ShiftScaleRotate, ElasticTransform
   - 결과 : 재출 횟수 1 날라감...fail.....
3. 세번째 - ElasticTransform 만 적용

   - encoder : resnext50
   - batch_size : 16
   - Augmentation 추가 : ElasticTransform
   - epoch : 30
   - 결과 : 0.6006
4. 네번째 - Augmentation 추가
   - encoder : resnext50
   - batch_size : 16
   - epoch : 40
   - Augmentation 추가 : ElasticTransform, GridDropout, GridDistortion
   - 결과 : 0.5759
5. 다섯번째
   - encoder : resnext50
   - batch_size : 16
   - epoch : 40
   - Augmentation 추가 : ElasticTransform
   - loss : (cross entropy * 0.9) + (focal loss * 0.1)
   - 결과 : 0.6150
6. 여섯번째
   - encoder : resnext50
   - batch_size : 16
   - epoch : 30
   - Augmentation 추가 : Elastic Transform, GlassBlur
   - 결과 : 0.5798

### 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

- 첫번째 se_resnext50모델의 경우에는 성능이 크게 나쁘지는 않았지만 모델을 불러오는 시간도 상당히 오래 걸리고 상대적으로 학습도 조금 느린 감이 없지 않아 있었습니다. 저는 resnext50을 기준으로 다양한 시도를 해볼 생각입니다.
- ElasticTransform은 사진에 왜곡을 주는 것 같습니다. 지금까지 실험 결과 성능을 가장 많이 올려 주었습니다.
- ElasticTransform, GridDropout, GridDistortion 3가지를 OneOf에 추가하여 활용했더니 점수가 하락했다.
- GridDropout의 단독사용은 성능을 떨어뜨렸습니다. 기준으로 잡은 resnext50만 사용한 모델의 성능보다도 낮은 성능을 보여줍니다. 다만 Elastic Transform과 GridDropout 2개를 조합해서 실험 해볼 예정입니다.
- GlassBlur도 성능을 하락 시켰습니다.
- 기존에 cross entropy loss만 사용했지만 이번에는 Focal loss를 섞어서 사용해 봤습니다.



## 210501

### 시도했던 내용

- 공통 setting : 
  - lr : 0.0001 
  - weight_decay : 1e-6
  - optimizer : Adam
  - seed : 21
  - batch_size : 16
  - Encoder weight : imagenet
  - model : DeepLabV3Plus
  - 기본 Augmentation : HorizonFliip, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion, normalize, ElasticTransform 을 사용합니다

1. 앞으로의 실험의 기준점을 잡기위해서 학습시키고 검증한 모델입니다.

   - encoder : resnext50

   - epoch : 30 
   - Resize(256, 256) 추가
   - LB : 0.5784

2. 어제 실험 해봤던 cross entropy loss + focal loss를 비교해보기 위해서 다시 실험 했습니다.

   - encoder : resnext50

   - epoch : 40 
   - Resize 
   - loss=cross entropy 0.9 + focal 0.1 
   - 0.5899

3. 

   - encoder : resnext50

   - epoch : 40 
   - Loss : Label smoothing 0.1 + cross entropy 0.9 
   -  Resize  추가
   - LB : 0.5801

4. 

   - encdoer : resnext50
   - epoch=40 
   - Resize /
   - Loss : Label smoothing 0.2 + cross entropy 0.7 + focal 0.1 
   - LB :  0.5902

5. 
   - encoder : efficientnetb4
   - batch=8 
   - epoch=30 
   - Loss : Focal 0.1 + Cross Entropy 0.9
   - LB :  0.6243



### 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

- Focal Loss를 제가 찾아보고 이해한 바로는 잘 찾은 class에 대해서는 loss를 적게 주고 잘 못찾은 class에 대해서는 loss를 크게 주는 것으로 알고 있습니다. 그래서 데이터가 불균형 할 때 자주 쓰인다고 제가 이해를 하고 있습니다. 그래서 cross entropy loss와 섞어서 사용해봤습니다. ( 단독으로 사용시에는 성능이 좋지 않습니다. 실험 해봤습니다 ㅎㅎ)
- 3번의 label smoothing loss는 모델의 과잉 확신을 방지할 수 있어 모델의 일반화 성능이 올려주는 loss로 이해 하였습니다. hard target을 soft target으로 바꿔주는 것입니다. 예를 들면 어떤 4개의 class를 예측한다고 했을 때 hard target이  [0, 1, 0, 0] 이라면 label smoothing을 실시하면 soft target [0.025, 0.925, 0,025, 0,025] 이렇게 변경해주는 loss인 것이죠. 수식을 이해하면 참 좋을텐데 수학이 약해서 그런 부분은 더 공부를 해야겠습니다...아무튼 이러한 점을 이용해서 저의 모델의 일반화 성능을 높여주기 위해서 cross entropy loss와 섞어서 사용했습니다. (물론 이것도 단독으로 사용하면 성능이 좋지 않습니다 ㅎㅎ)
- 4번 실험에서는 3가지의 loss를 섞어서 학습 시켜봤습니다. 저도 길희님 처럼 어떤 비율이 좋다 라고는 말씀을 못드리겠습니다만 확실히 3가지를 섞어서 사용한 모델이 성능이 미미하지만 더 좋게 나왔습니다.
- 5번째 실험은 길희님의 추천으로 efficientNet b4 를 encoder로 사용해 봤는데 성능이 아주 좋게 나왔습니다. 다만 학습속도가 너무 오래 걸려서 이걸로 실험을 하기보다는 좀 더 빠른 모델로 실험을 하고 마지막 제출하기 전에 학습해서 검증할 때 사용하려고 합니다 ㅎㅎ
- 내일은 Dice Loss를 사용해보려고 합니다. 객체의 경계선을 잘 인식할 수 있는? 그런 loss라고 대충 이해했는데 확실히 논문 읽기가 어렵네요..혹시 제가 잘 못 이해 했다면 피어세션때 알려주시면 감사하겠습니다 ㅎㅎ



## 210502

### 시도했던 내용

- 공통 setting : 
  - lr : 0.0001 
  - weight_decay : 1e-6
  - optimizer : Adam
  - seed : 21
  - batch_size : 16
  - Encoder weight : imagenet
  - model : DeepLabV3Plus
  - 기본 Augmentation : HorizonFliip, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion, normalize, ElasticTransform, Resize(256, 256) 을 사용합니다

1. encoder : efficientnetb4
   - batch_size : 8 
   - epoch : 40 
   - loss : Label smoothing(0.2) + cross entropy(0.7) + focal(0.1)
   - LB : 0.6243

2. encoder : resnext50 
   - epoch : 40 
   - loss : Label smoothing(0.3) + cross entropy(0.5) + focal(0.2) 
   - LB : 0.5700

3. encoder : resnext50 
   - epoch : 40
   - loss : cross entropy(0.9 )+ Dice(0.1)
   - LB : 0.5737

4. encoder : resnext50
   - epoch : 40 
   - loss : Label smoothing(0.1) + cross entropy(0.8) + focal(0.1) 
   - LB : 0.5864

5. encoder : resnext50) 
   - epoch : 40 
   -  loss : Dice(0.2) + cross entropy(0.7) + focal(0.1) 
   - LB : 0.5769



### 결과를 보고 난 나의 견해 & 다음에 시도해 볼 것

- 어제 실험했던 내용 중 Label smoothing loss, Cross entropy loss, Focal loss를 2 : 7 : 1 로 섞어서 학습한 모델의 성능이 높게 나와서 efficientNet b4에도 시도해 보았다. 결과가 Focal loss 와 Cross entropy loss를 1 : 9 로 섞어서 학습한 모델과 똑같이 나왔습니다. 사실 어제 실험에서도 두개의 성능 차이가 그렇게 크지 않았습니다.
- 2번 실험은 모델의 일반화 성능을 높여주고 싶어서 비율을 변경하고 실험했습니다.
- 객체의 boundary를 더 잘 인식할 수 있는 Dice loss는 전반적으로 결과가 이전의 Focal loss나 Label smoothing loss를 섞어 사용한 모델들 보다 성능이 낮게 나왔습니다.
- Label smoothing loss의 비율을 높여 일반화 성능을 올리면 모델의 성능이 좋아지지 않을까 하고 생각을 했었는데 오히려 낮아졌습니다. 그 이유에 대해서 저의 개인적인 생각으로는 Label smoothing loss는 hard target을 soft target으로 바꾸어주는데 Label smoothing loss의 비율이 높아 오히려 분류해야 하는데 class의 경계가 모호해져서 잘 분류를 못하는 것이 아닐까? 하고 생각하고 있습니다. 혹시 저와 다른 견해가 있으시 거나 잘 이해하신 분이 있다면 말씀해주시면 감사하겠습니다.
- Loss의 최적의 비율을 계속해서 실험하고 싶으나 시간을 너무 많이 소모할 것 같아서 내일 부터는 다른 실험을 해보려고 합니다. 일단 우선적으로 optimizer를 Adam에서 AdamP로 바꾸어서 학습, lr_scheduler를 이용해서 학습, 새로운 Augmentation인 GridMask를 적용해볼 예정입니다. 3가지 실험을 결과가 전부 기준보다 높게 나온다면 섞어서 사용해볼 예정입니다.
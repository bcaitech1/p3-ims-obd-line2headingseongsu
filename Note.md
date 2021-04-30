# '성수행 2호선 열차' Daily Note

## 2021.04.27
* 원승재
  - 원래 cutmix에 대해서 실험하려고 했으나 당장 코드보고 적용이 어려워서 일단 베이스라인 코드 익히고 실험 돌려봄
  - Resize(256), HorizontalFlip, RandomRotate90, VerticalFlip 총 4가지 적용해서 베이스라인 모델로 실험 진행
  - Normalize 적용해보려고 했으나, cv2로 출력했을 떄 이미지가 제대로 보이지 않아서 내일 같이 얘기해 볼 예정. 아마 학습할 땐 지장이 없고 cv2 출력 문제일 거라 생각됨
* 윤석진
  - 다른 annotation인데 image_id와 category_id가 같은 경우 : 한 사진에 똑같은 종류의 여러 객체가 존재하는 경우
  - 한 annotation의 segmentation 리스트 안의 리스트가 여럿인 경우 : 하나의 객체가 앞의 또다른 객체에 의해 가려져 segmentation상 둘 이상으로 보이는 경우
  - 자세한 설명과 사진 자료는 SeokJin 폴더에 올림, 그 외에 베이스라인 코드를 이해함, visual code의 플롯을 파일로 저장할 때 pdf로 저장하면 왜 안 열리는지, 바로 로컬에 저장하면 왜 파일이 안 생기는지 모르겠다.
* 이민우
  - Efficientunet-b7 으로 실험. CLAHE, Input shape resize를 진행해보았으나 성능은 내려감.
  - 그런데 시각적으로는 CLAHE를 사용했을 때 객체를 보다 잘 검출해내고 Loss도 보다 많이 떨어짐. epoch 조절이 필요할 것으로 예상.
  - 자세한 사항은 MWLee/실험.txt에 작성
* 조범경
  - deeplab v1으로 daily 미션 부분 작성, 실행해봄. 매끄럽지 못한 부분 있었지만 mIoU 점수 0.3452 인걸 보니 잘 돌아가기는 한듯...?  
  - 학습시간이 FCN보다 훨씬 길었던 것을 감안하면 비용대비 결과는 후진걸로....
* 최길희
  - 속도가 너무 느려서 수렴이 빠른 AdamP를 쓰고 epoch를 줄이려고 실험해봄 -> 성능 낮음=실패
  - 속도 느린걸 해결하려고 256*256 으로 이미지 resize -> 성능은 떨어지나 시간은 1/3로 떨어짐
  - 256*2556 성능 올려보려고 lr_scheduler 써봄 -> 더 떨어짐 = 실패
  - 256*256의 이미지에서 전처리 실험
    + HorizontalFlip, VerticalFlip, RandomRotate90 : 성능 오름
    + MotionBlur, OpticalDisortion, GaussNoise : 성능이 크게 오름
    + 날씨 Effect들(RandomRain, RandomSnow, RandomSunFlare, RandomFog) : 애매함
    + Color 관련 (RGBShift, HueSaturationValue, ChannelShuffle) : LB에서 확인은 못했지만 Validation 성능 오름
* 황훈
  - 시도
    + 26일 daily_mission 에서 fcn16s를 pretrain된 VGG16을 이용하여 구현하였고 학습을 돌렸다. 너무느려서 epoch은 6만 돌려주었다. 결과 : 0.3218
    + 27일 daily_mission 에서 sgenet을 완성하여 학습을 돌려보았다. epoch : 14 결과 : 0.3075
    + torchvision에서 제공해주는 segmantic segmetaion 모델 DeepLabV3_resnet50 과 resnet101을 시도해보았다. CUDA 메모리 부족 현상이 발생... 좀 더 고민해 봐야 할 것 같다.
  - pretrain된 모델을 사용하는 것이 성능이 더 좋은 것 같다. 
  - segNet 도 생각보다 너무 느리다...
## 2021.04.28
* 최길희
  - 256*256의 이미지에서 전처리 실험(이어서)
    + 밝기 관련 transform (RandomContrast, RandomGamma, RandomBrightness) : 애매
    + CLAHE : 애매 + 사용하려면 baseline code 수정 필요
    + # 전날의 transform을 모두 통합하여 사용하면 LB 0.04오름 #
  - CUDA out of memory 오류
    + transform과 관련이 없었다. 서버가 memory를 해제하지 않아 생긴 문제로 서버를 새로 생성했다.
    + 지속적으로 GPU의 memory를 모니터링하고 cache를 해제해주자!
    + batch_size 16으로 줘도 전혀 문제 없다!
  - torchvision의 fcn_resnet50 사용
    + LB : 0.4956 -> 성능이 뛰어나다
    + 토론 게시판을 참고하면 아무 transform도 하지 않았을 때 0.45나온다고 했으니 내 점수는 온전히 transform에 의해 오른 점수이다.
  - ToFloat 대신 Normalize 사용 & best model 저장 기준 mIoU로 설정
    + best model 저장 기준 mIoU로 설정 : 이건 단순히 몇번 EPOCH의 모델을 쓸지 판단하는데만 쓰이는 것이므로 모델의 성능과는 관련이 없다. 단지 이걸 쓰니 더 높은 LB의 epoch를 잘 선택한다는 것을 확인할 수 있었다.
    + ToFloat 대신 Normalize 사용 : 이전의 모델과 비교하여 성능이 올랐다.
* 이민우
  - unet 사용 포기 및 deeplabv3_resnet101 사용. 성능 많이 오름.
  - 256*256 전처리 및 Normalize 수행 => 성능 크게 오름
  - 수평, 수직, 회전으로 데이터 네 배로 증강 후 실행 => 아주 약간 오름
* 황훈
  - torchvision DeepLabV3를 학습할 떄 마다 CUDA memory오류가 났지만 해결
    + 학습 후에는 Kernel을 종료해야 CUDA memory 사용이 초기화 된다.
  - DeepLabV3Plus에 Xception모델을 backbone으로 사용하려 했지만 Xception 코드에서 dilation을 거부하여 실패하였다.
    + segmentation_models_pytorch 라는 모듈을 사용하였다.
    + 원작자 github에도 질문이 올라와 있지만 해결 방법은 아직까지 없는 것 같다.
  - DeepLabV3_ResNet50을 시도했지만 성능이 너무 낮게 나왔다.
    + dataset을 전처리하는 부분에서 픽셀 값이 커지면 계산하는데 어려움을 줄 수 있어서 255로 나누어 정규화하는 부분을 제거하고 normalize를 시도했어야 했는데 제거하지 않고 학습을 하여 성능이 낮게 나온 것으로 추정 현재 수정 후 다시 학습 시도 중.
## 2021.04.29
* 이민우
  - deeplabv3+_resnet101_32x4 사용. 성능 오름.
  - 전처리에는 변화를 주지 않음. 그대로 네 배 증강 후 랜덤 blur 작업 수행. (성능은 잘 오르지만 시간이 오래 걸림.)

* 황훈

  - 공통 setting : 
    - lr = 0.0001 
    - weight_decay = 1e-6
    - optimizer = Adam
    - loss = cross entropy
    - normalize 적용
    - seed = 21
    - Encoder weight : imagenet

  - model : DeeplabV3
     - backbone : resnet101 
     - Resize(256,256)
     - batch=24 
     - 결과 : 0.5070 

  - model : DeeplabV3Plus
     - backbone : resnet101
     - Augmentation : Resize(256,256)  
     - batch=24 
     - 0.5522

  - model : DeeplabV3Plus
     - resnet101
     - Augmentation : Resize(256,256), Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion
     - batch=24 
     - seed=42 
     - 결과 : 0.5553

  - model : DeeplabV3Plus
     - backbone : resnet101
     - Augmentation : (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
     - batch=4
     - 결과 :  0.5027
     - 학습 Validation mIoU: 0.4954

  - model : DeeplabV3Plus
     - backbone : resnet50 
     - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
     - batch=16
     - 결과 : 0.5817

  - model : DeeplabV3Plus
     - backbone : resnext50
     - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
     - batch=16 
     - 결과 : 0.5881
     - Validation mIoU: 0.5546

  - model : U-net 
     - backbone : EfficeintNet b4
     - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
     - batch=8 
     - encoder_weight : noisy_study
     - 결과 :  0.5022 
  - model :  U-net 
     - backbone : EfficientNet b0
     - Augmentation :  (Horizon, VerticalFlip, RandomRotate90, MotionBlur, GaussianBlur, OpticalDistortion)
     - batch=16 
     - encoder_weight : noisy_study 
     - epoch=40 / 0.4757
  - 자세한 내용은 폴더안에 일지에 담겨져 있습니다.
* 최길희
  - 1)model : DeeplabV3
    + Backbone : Resnet50
    + 학습시간 : 6시간
    + mIoU : 0.5379
    + LB : 0.5781
  - 2)model : DeeplabV3Plus
    + Backbone : Resnet50
    + 학습시간 : 1시간
    + mIoU : 0.5250
    + LB : 0.5628
    + 1번 모델과의 비교 : 성능은 조금 차이나는 것에 비해 학습시간은 DeepLabV3Plus가 월등히 적게 든다
  - 3)model : Unet++
    + Backbone : Resnet50
    + 학습시간 : 3시간 13분
    + mIoU : 0.3322
    + 결론 : 쓰지마세요 쓰레깁니다
  - 4)model : DeeplabV3Plus
    + Backbone : Resnet101
    + 학습시간 : 1시간 26분
    + mIoU : 0.5227
    + LB : 0.5911
    + 결론 : 성능도 좋고 속도도 빠릅니다!이겁니다!
  - 5)model : DeeplabV3Plus
    + Backbone : resnext50_32x4d
    + 학습시간 : 1시간 30분
    + mIoU : 0.5594
    + LB : 0.5882
    + 결론 : 얘도 성능이 준수합니다. 하지만 4번 모델이 조금 더 좋군여
  - 6)model : DeeplabV3Plus
    + Backbone : densenet161
    + 결론 : densenet161도 그 오류로 사용 못합니다
  - 7)model : DeeplabV3Plus
    + Backbone : efficientnet-b0
    + 학습시간 : 2시간 30분
    + mIoU : 0.5272
    + 결론 : Efficient-b0인데도 성능이 이러면 더 높은 애들은 더 좋지 않을까 생각합니다. 단 Efficient 계열은 메모리 차지가 아주 큽니다 Batchsize 줄여야함

  *여기서 부터는 loss를 좀 바꿔봤습니다*
  - 비교용 모델
    + model : DeeplabV3Plus
    + Backbone : resnet50
    + mIoU : 0.5250
    + LB : 0.5628
    + Loss : CrossEntropyLoss
    + 사용이유 : 일단 학습시간이 빠르면서 성능도 준수해서 이걸 비교용 모델로 사용했습니다.
  - 1)FocalLoss(gamma=2)
    + model : DeeplabV3Plus
    + Backbone : resnet50
    + mIoU : 0.2548
    + Loss : FocalLoss(gamma=2)
    + 결론 : 쓰레깁니다. 단독으로 사용하지 마세요.
  - 2)CustomLoss1
    + model : DeeplabV3Plus
    + Backbone : resnet50
    + mIoU : 0.5295
    + LB : 0.5779
    + Loss : CrossEntropyLoss*0.8 + FocalLoss(gamma=2)*0.2
    + 결론 : CrossEntropy만 쓴 것보다 성능이 조금 더 올랐습니다.

## 2021.04.30

- 황훈

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
     - epoch : 30
     - Augmentation 추가 : ElasticTransform
     - 결과 : 0.6006
  4. 네번째 - Augmentation 추가
     - encoder : resnext50
     - batch_size : 16
     - epoch : 40
     - Augmentation 추가 : ElasticTransform, GridDropout, GridDistortion
     - 결과 : 0.5759
  5. 결과를 통해 알 수 있었던 내용 & 다음에 시도해 볼 것

     - 첫번째 se_resnext50모델의 경우에는 성능이 크게 나쁘지는 않았지만 모델을 불러오는 시간도 상당히 오래 걸리고 상대적으로 학습도 조금 느린 감이 없지 않아 있었습니다. 저는 resnext50을 기준으로 다양한 시도를 해볼 생각입니다.
     - ElasticTransform 같은 경우에는 밑에 사진과 같이 약간 사진에 왜곡을 주는 것 같습니다.
     - 

  


## 적용은 못했지만 Idea는 있다
* VGG16 대신 ResNet, EfficientNet으로 백본 교체
* 지금 Baseline Architecture는 FCN-8s인데 강의에 나온 Architecture의 성능은 다음과 같다. 더 좋은 Architecture를 사용해보는 건 어떨까?
![Segmentation Architecture별 성능](./img/Segmentation_Architecture.png)
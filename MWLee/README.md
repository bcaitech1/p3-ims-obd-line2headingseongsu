## 05/03 월
#### 1.

모델 : deeplabv3+resnext101_32x4d

전처리 : Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss

batch_size : 16 (사실상 32)

epochs : 17

random_seed : 42

정확도 : 0.6426

학습 시간 (1epoch) : 15~17m

비고 :

    SterLR 버리고 CosineAnnealingLR 사용.

    처음에는 엄청 잘 되다가, 어느 순간부터 다시 정체되기 시작.

    성능이 오르긴 올랐는데, 막 좋다는 생각은 잘 안듬.
    

#### 2.

모델 : deeplabv3+resnext101_32x4d

전처리 : Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss


batch_size : 16 (사실상 32)

epochs : 17

random_seed : 42

정확도 : 0.6352

학습 시간 (1epoch) : 15~17m

비고 :

    CosineAnnealingLR 버리고 ReduceLROnPlateau 사용.

    오히려 더 떨어짐 (별로임)



#### 3.

모델 : deeplabv3+resnext101_32x4d

전처리 : Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss(길희님 실험 결과 가장 좋은 loss)

batch_size : 16 (사실상 32)

epochs : 11

random_seed : 42

정확도 : 0.6022

학습 시간 (1epoch) : 15~17m

비고 :

    CosineAnnealingLR 사용.

    길희님 실험 결과 중 가장 좋은 Loss 조합을 사용

    떨어짐. 내 코드랑은 안맞는듯.



#### 4.

모델 : deeplabv3+resnext101_32x4d

전처리 : -

batch_size : 16 (사실상 32)

epochs : -

random_seed : 42

정확도 : 0.6528

학습 시간 (1epoch) : -

비고 :

    0.6426, 0.6382, 0.6358 SoftVote 앙상블

    생각만큼 큰 폭 변경은 아니지만 좋음.

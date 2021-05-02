## 05/02 일

- 개인적으로 eff_b4는 좋은 성능을 못보이는 중.
- adamP 사용 시 미세한 성능 증가 확인
- 256 Resize 제거 시 성능 대폭 증가 확인

#### 1.

모델 : deeplabv3+efficientnet_b4

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur / Ela+grid+optical / 두 개 합친 거) => 4배로 증강 / CustomLoss

batch_size : 8 (사실상 16)

epochs : 12

random_seed : 42

정확도 : 0.5665

학습 시간 (1epoch) : 30~31m

비고 :

    무슨 차이인지는 모르겠지만, 내 코드에 eff는 어울리지 않는 것 확정.

#### 2.

모델 : deeplabv3+resnext101_32x4d

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss

batch_size : 16 (사실상 32)

epochs : 17

random_seed : 42

정확도 : 0.6103

학습 시간 (1epoch) : 8~9m

비고 :

    황훈님의 말씀대로 배치 사이즈가 모델에 영향을 미침을 확인.
   

#### 3.

모델 : deeplabv3+resnext101_32x4d

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss

batch_size : 16 (사실상 32)

epochs : 14

random_seed : 42

정확도 : 0.6104

학습 시간 (1epoch) : 8~9m

비고 :

    adamP 사용. 파라미터는 그대로 줌.

    성능 미세하게 향상

    혹시 몰라 20epoch까지 돌린 모델도 사용해보았으나 과적합.
    

#### 4.

모델 : deeplabv3+resnext101_32x4d

전처리 : Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss

batch_size : 16 (사실상 32)

epochs : 18

random_seed : 42

정확도 : 0.6382 ★

학습 시간 (1epoch) : 16~17m

비고 :

    adamP 사용 (파라미터 그대로)

    Resize를 안하면 결과가 어떨까 해서 사용해봄.

    대폭 상승 확인. => 최종 제출은 무조건 512*512로 학습할 것
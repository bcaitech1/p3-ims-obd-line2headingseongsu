## 4/27 화

- ※모든 모델들이 epoch 9에서 가장 좋은 성능을 보임 (valid set 기준)
- ※하지만 그냥 epoch 10까지 전부 한 모델들 사용

#### 1.

모델 : deeplabv3+resnext101_32x4d

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur / Ela+grid+optical / 두 개 합친 거) => 4배로 증강 / CustomLoss

batch_size : 4 (사실상 16)

epochs : 19

random_seed : 42

정확도 : 0.5929

학습 시간 (1epoch) : 16~17m

비고 :

    epoch의 차이인지, augmentation의 차이인지는 몰라도 성능이 조금 오름.

    epoch를 더 주면 더 오를 것 같음


#### 2.

모델 : deeplabv3+resnext101_32x4d

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur / Ela+grid+optical / 두 개 합친 거) => 4배로 증강 / CustomLoss

batch_size : 4 (사실상 16)

epochs : 39

random_seed : 42

정확도 : 0.5987

학습 시간 (1epoch) : 16~17m

비고 :

    Valid_acc와 관계 없이 epoch를 더 줬더니 오름.

    
#### 2.

모델 : deeplabv3+resnext101_32x4d

전처리 : Resize(256), Normalize, Augmentation (Hor+Ver+Rot+Blur+Ela+grid+optical) => 2배로 증강 / CustomLoss

batch_size : 4 (사실상 16)

epochs : 16

random_seed : 42

정확도 : 0.5988 ★

학습 시간 (1epoch) : 8~9m

비고 :

    성능이 올라가지 않는 이유가, 비록 Augmentation을 했다고는 하나 결국에는 동일 이미지를 계속 학습시켜서가 아닐까 유추.

    그래서 욕심을 버리고 두 배로 증강시켰더니 바로 오름.
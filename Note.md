# '성수행 2호선 열차' Daily Note

## 2021.04.27
* 원승재
  - 원래 cutmix에 대해서 실험하려고 했으나 당장 코드보고 적용이 어려워서 일단 베이스라인 코드 익히고 실험 돌려봄
  - Resize(256), HorizontalFlip, RandomRotate90, VerticalFlip 총 4가지 적용해서 베이스라인 모델로 실험 진행
  - Normalize 적용해보려고 했으나, cv2로 출력했을 떄 이미지가 제대로 보이지 않아서 내일 같이 얘기해 볼 예정. 아마 학습할 땐 지장이 없고 cv2 출력 문제일 거라 생각됨
* 윤석진
  - 한 일 쓰기!
* 이민우
  - Efficientunet-b7 으로 실험. CLAHE, Input shape resize를 진행해보았으나 성능은 내려감.
  - 그런데 시각적으로는 CLAHE를 사용했을 때 객체를 보다 잘 검출해내고 Loss도 보다 많이 떨어짐. epoch 조절이 필요할 것으로 예상.
  - 자세한 사항은 MWLee/실험.txt에 작성
* 조범경
  - deeplab v1으로 daily 미션 부분 작성, 실행해봄. 매끄럽지 못한 부분 있었지만 mIoU 점수 0.3452 인걸 보니 잘 돌아가기는 한듯...?  
  - 학습시간이 FCN보다 훨씬 길었던 것을 감안하면 비용대비 결과는 후진걸로....
* 최길희
  - 한 일 쓰기!
* 황훈
  - 한 일 쓰기!

* * *
## 적용은 못했지만 Idea는 있다
* VGG16 대신 ResNet, EfficientNet으로 백본 교체
* 지금 Baseline Architecture는 FCN-8s인데 강의에 나온 Architecture의 성능은 다음과 같다. 더 좋은 Architecture를 사용해보는 건 어떨까?
![Segmentation Architecture별 성능](./img/Segmentation_Architecture.png)

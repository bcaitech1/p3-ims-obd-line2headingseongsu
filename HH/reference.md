## Segmentatioin Models
- https://smp.readthedocs.io/en/latest/index.html
- https://github.com/qubvel/segmentation_models.pytorch
- 이 라이브러리를 이용해서 pre-trained 모델을 사용했습니다.
- segmentation model과 backbone도 모델도 이것 저것 시도해보기 편한것 같습니다.

```python
import segmentation_models_pytorch as smp


model = smp.DeepLabV3Plus(encoder_name="resnet101",       # encoder에서 사용할 모델을 선택하는 파라미터 입니다 공홈에 가시면 더 많은 모델을 찾아 볼 수 있습니다.
                          encoder_weights="imagenet",     # imagenet을 사용해서 pre-train 된 weight를 불러오는 파라미터 입니다.
                          classes=12)                     # model output을 설정하는 파라미터 입니다.

                                                          # 이외에도 많은 파라미터가 있습니다 자세한 건 공홈이나 github에서 확인해 주세요.
```


### 🛠 Installation <a name="installation"></a>
PyPI version:
```bash
$ pip install segmentation-models-pytorch
````

- 저는 Xception을 시도해보려고 했는데 Dilation 문제가 발생해서 못했는데 여기저기 찾아봐도 해결책이 없네요. 일단 지금은 resnet101을 해봤는데 성적이 꽤 잘 나옵니다. paperwithcode나 논문을 찾아보면서 더 좋은 백본 모델이 있나 찾아보거나 아님 Augmentation을 더 시도해보는 방향으로 저는 생각중입니다.

- 많은 도움이 되었으면 좋겠습니다~ ^^7


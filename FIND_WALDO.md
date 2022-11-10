# FIND WALDO🔍

## CNN을 활용하여 월리 찾기!
어릴 적 월리를 찾아라를 해본적이 있나요? 무수히 많은 사람들 중에서 월리를 찾는데 다들 얼마나 걸리셨나요?  
<br> </br>
## 
## Import Package
### numpy 로드
```
import numpy as np
```
### 딥러닝을 keras 로드
callback 함수를 이용하여 모델을 훈련시키는 동안 발생하는 이벤트를 원하는 동작으로 수행시킨다. 
```
import keras.layers as keras
import keras.optimizers as optimizers
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau 
```
### 이미지 처리를 위한 PIL, skimage 로드

```
from PIL import Image
from skimage.transform import resize
```
### 시각화를 위한 matplotlib, seaborn 로드
```
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```
<br> </br>
## Load Dataset
### npy형태인 데이터 4개를 가져온다.
### 텐서를 사용하는 keras는 float형태로 바꾸어야 한다.
### 255로 나누는 이유는 픽셀의 범위는 0~255인데, 0~1로 rescale하여 사용하여야 한다.

```
imgs = np.load('dataset/imgs_uint8.npy').astype(np.float32) / 255.
labels = np.load('dataset/labels_uint8.npy').astype(np.float32) / 255.
waldo_sub_imgs = np.load('dataset/waldo_sub_imgs_uint8.npy', allow_pickle=True) / 255.
waldo_sub_labels = np.load('dataset/waldo_sub_labels_uint8.npy', allow_pickle=True) / 255.
```
### 가져온 데이터의 정보를 출력해보자
```
print(imgs.shape, labels.shape)
print(waldo_sub_imgs.shape, waldo_sub_labels.shape) 
```
### 출력 결과
### 컬러인 2800*1760 사이즈의 18개의 사진 데이터가 담겨져있다.
```
(18, 1760, 2800, 3) (18, 1760, 2800)
(18,) (18,)
```


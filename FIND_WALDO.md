# FIND WALDO๐

## CNN์ ํ์ฉํ์ฌ ์๋ฆฌ ์ฐพ๊ธฐ!
์ด๋ฆด ์  ์๋ฆฌ๋ฅผ ์ฐพ์๋ผ๋ฅผ ํด๋ณธ์ ์ด ์๋์? ๋ฌด์ํ ๋ง์ ์ฌ๋๋ค ์ค์์ ์๋ฆฌ๋ฅผ ์ฐพ๋๋ฐ ๋ค๋ค ์ผ๋ง๋ ๊ฑธ๋ฆฌ์จ๋์?  
<br> </br>
## 
## Import Package
### numpy ๋ก๋
```
import numpy as np
```
### ๋ฅ๋ฌ๋์ keras ๋ก๋
callback ํจ์๋ฅผ ์ด์ฉํ์ฌ ๋ชจ๋ธ์ ํ๋ จ์ํค๋ ๋์ ๋ฐ์ํ๋ ์ด๋ฒคํธ๋ฅผ ์ํ๋ ๋์์ผ๋ก ์ํ์ํจ๋ค. 
```
import keras.layers as keras
import keras.optimizers as optimizers
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau 
```
### ์ด๋ฏธ์ง ์ฒ๋ฆฌ๋ฅผ ์ํ PIL, skimage ๋ก๋

```
from PIL import Image
from skimage.transform import resize
```
### ์๊ฐํ๋ฅผ ์ํ matplotlib, seaborn ๋ก๋
```
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```
<br> </br>
## Load Dataset
### npyํํ์ธ ๋ฐ์ดํฐ 4๊ฐ๋ฅผ ๊ฐ์ ธ์จ๋ค.
### ํ์๋ฅผ ์ฌ์ฉํ๋ keras๋ floatํํ๋ก ๋ฐ๊พธ์ด์ผ ํ๋ค.
### 255๋ก ๋๋๋ ์ด์ ๋ ํฝ์์ ๋ฒ์๋ 0~255์ธ๋ฐ, 0~1๋ก rescaleํ์ฌ ์ฌ์ฉํ์ฌ์ผ ํ๋ค.

```
imgs = np.load('dataset/imgs_uint8.npy').astype(np.float32) / 255.
labels = np.load('dataset/labels_uint8.npy').astype(np.float32) / 255.
waldo_sub_imgs = np.load('dataset/waldo_sub_imgs_uint8.npy', allow_pickle=True) / 255.
waldo_sub_labels = np.load('dataset/waldo_sub_labels_uint8.npy', allow_pickle=True) / 255.
```
### ๊ฐ์ ธ์จ ๋ฐ์ดํฐ์ ์ ๋ณด๋ฅผ ์ถ๋ ฅํด๋ณด์
```
print(imgs.shape, labels.shape)
print(waldo_sub_imgs.shape, waldo_sub_labels.shape) 
```
### ์ถ๋ ฅ ๊ฒฐ๊ณผ
### ์ปฌ๋ฌ์ธ 2800*1760 ์ฌ์ด์ฆ์ 18๊ฐ์ ์ฌ์ง ๋ฐ์ดํฐ๊ฐ ๋ด๊ฒจ์ ธ์๋ค.
```
(18, 1760, 2800, 3) (18, 1760, 2800)
(18,) (18,)
```


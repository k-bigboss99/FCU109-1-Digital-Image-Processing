# 109-1數位影像處理
## 環境安裝
- numpy
```python=
import numpy        # 載入numpy
import numpy as np  # 載入numpy，命名空間為np
form numpy import*  # 全部載入
```
```
IMREAD_UNCHANGE(-1) : 根據原始影像的型態讀取
IMREAD_GRAYSCALE(0) : 讀取為灰階影像
IMREAD_COLOR(1) : 讀取為色彩影像
IMREAD_ANYDEPTH(2) : 讀取任意位元深度的影像
IMREAD_ANYCOLOR(4) : 讀取任意色彩的影像
```
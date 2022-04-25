# My TimeSeries study code
### 这个仓库我用来学习timeSeries相关的知识。
### 1.Informer-aoweichwn
#### 这是我浮现的第一篇timeseries相关的论文，大致原理都在论文里说的很清楚了，这里只写一下代码如何使用：
#####    直接下载代码，在终端中打开main_informer.py
#####    输入以下代码即可：
    ```python
    #!/usr/bin/env python3
    # ETTh1
    python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

    # ETTh2
    python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

    # ETTm1
    python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
    ```
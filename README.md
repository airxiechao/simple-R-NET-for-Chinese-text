# simple-R-NET-for-Chinese-text
简单中文R-NET问答模型

准备数据
---

* paragraph.json：问答数据，格式为```[{"paragraph":"段落文字", "qas":[{"question":"问题", "start": 答案起始位置, "end": 答案结束位置}]}]```

* wiki.zh.vec：中文向量字典（https://fasttext.cc/docs/en/pretrained-vectors.html )，下载到代码目录

训练和测试
---
运行 ```python rnet.py```，开始训练，训练结束后执行测试，输入一段文字和问题，会输出模型做出的回答


参考
---

1、https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

2、https://github.com/unilight/R-NET-in-Tensorflow

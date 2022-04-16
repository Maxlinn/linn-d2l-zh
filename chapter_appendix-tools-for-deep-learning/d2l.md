# `d2l` API 文档
:label:`sec_d2l`

`d2l`包以下成员的实现及其定义和解释部分可在<img src="https://github.com/d2l-ai/d2l-en/tree/master/d2l" alt="源文件">中找到。


```eval_rst
.. currentmodule:: d2l.torch
```


## 模型

```eval_rst 
.. autoclass:: Module
   :members: 

.. autoclass:: LinearRegressionScratch
   :members:

.. autoclass:: LinearRegression
   :members:    

.. autoclass:: Classification
   :members:
```

## 数据

```eval_rst 
.. autoclass:: DataModule
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: FashionMNIST
   :members:
```

## 训练

```eval_rst 
.. autoclass:: Trainer
   :members: 

.. autoclass:: SGD
   :members:
```

## 公用

```eval_rst 
.. autofunction:: add_to_class

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: num_gpus

.. autoclass:: ProgressBoard
   :members: 

.. autoclass:: HyperParameters
   :members:
```


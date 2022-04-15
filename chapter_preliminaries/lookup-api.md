# 查阅文档


由于本书篇幅限制，我们不可能介绍每一个PyTorch函数和类（你可能也不希望我们这样做）。
API文档、其他教程和示例提供了本书之外的大量文档。
在本节中，我们为你提供了一些查看PyTorch API的指导。


## 查找模块中的所有函数和类

为了知道模块中可以调用哪些函数和类，我们调用`dir`函数。
例如，我们可以(**查询随机数生成模块中的所有属性：**)



```python
import torch

print(dir(torch.distributions))
```

    ['AbsTransform', 'AffineTransform', 'Bernoulli', 'Beta', 'Binomial', 'CatTransform', 'Categorical', 'Cauchy', 'Chi2', 'ComposeTransform', 'ContinuousBernoulli', 'CorrCholeskyTransform', 'Dirichlet', 'Distribution', 'ExpTransform', 'Exponential', 'ExponentialFamily', 'FisherSnedecor', 'Gamma', 'Geometric', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Independent', 'IndependentTransform', 'Kumaraswamy', 'LKJCholesky', 'Laplace', 'LogNormal', 'LogisticNormal', 'LowRankMultivariateNormal', 'LowerCholeskyTransform', 'MixtureSameFamily', 'Multinomial', 'MultivariateNormal', 'NegativeBinomial', 'Normal', 'OneHotCategorical', 'OneHotCategoricalStraightThrough', 'Pareto', 'Poisson', 'PowerTransform', 'RelaxedBernoulli', 'RelaxedOneHotCategorical', 'ReshapeTransform', 'SigmoidTransform', 'SoftmaxTransform', 'StackTransform', 'StickBreakingTransform', 'StudentT', 'TanhTransform', 'Transform', 'TransformedDistribution', 'Uniform', 'VonMises', 'Weibull', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'bernoulli', 'beta', 'biject_to', 'binomial', 'categorical', 'cauchy', 'chi2', 'constraint_registry', 'constraints', 'continuous_bernoulli', 'dirichlet', 'distribution', 'exp_family', 'exponential', 'fishersnedecor', 'gamma', 'geometric', 'gumbel', 'half_cauchy', 'half_normal', 'identity_transform', 'independent', 'kl', 'kl_divergence', 'kumaraswamy', 'laplace', 'lkj_cholesky', 'log_normal', 'logistic_normal', 'lowrank_multivariate_normal', 'mixture_same_family', 'multinomial', 'multivariate_normal', 'negative_binomial', 'normal', 'one_hot_categorical', 'pareto', 'poisson', 'register_kl', 'relaxed_bernoulli', 'relaxed_categorical', 'studentT', 'transform_to', 'transformed_distribution', 'transforms', 'uniform', 'utils', 'von_mises', 'weibull']


通常，我们可以忽略以“`__`”（双下划线）开始和结束的函数（它们是Python中的特殊对象），
或以单个“`_`”（单下划线）开始的函数（它们通常是内部函数）。
根据剩余的函数名或属性名，我们可能会猜测这个模块提供了各种生成随机数的方法，
包括从均匀分布（`uniform`）、正态分布（`normal`）和多项分布（`multinomial`）中采样。

## 查找特定函数和类的用法

有关如何使用给定函数或类的更具体说明，我们可以调用`help`函数。
例如，我们来[**查看张量`ones`函数的用法。**]



```python
help(torch.ones)
```

    Help on built-in function ones:
    
    ones(...)
        ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
        
        Returns a tensor filled with the scalar value `1`, with the shape defined
        by the variable argument :attr:`size`.
        
        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.
        
        Keyword arguments:
            out (Tensor, optional): the output tensor.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if ``None``, uses the current device for the default tensor type
                (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
                for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
        
        Example::
        
            >>> torch.ones(2, 3)
            tensor([[ 1.,  1.,  1.],
                    [ 1.,  1.,  1.]])
        
            >>> torch.ones(5)
            tensor([ 1.,  1.,  1.,  1.,  1.])
    


从文档中，我们可以看到`ones`函数创建一个具有指定形状的新张量，并将所有元素值设置为1。
让我们来[**运行一个快速测试**]来确认这一解释：



```python
torch.ones(4)
```




    tensor([1., 1., 1., 1.])



在Jupyter记事本中，我们可以使用`?`指令在另一个浏览器窗口中显示文档。
例如，`list?`指令将创建与`help(list)`指令几乎相同的内容，并在新的浏览器窗口中显示它。
此外，如果我们使用两个问号，如`list??`，将显示实现该函数的Python代码。

## 小结

* 官方文档提供了本书之外的大量描述和示例。
* 我们可以通过调用`dir`和`help`函数或在Jupyter记事本中使用`?`和`??`查看API的用法文档。

## 练习

1. 在深度学习框架中查找任何函数或类的文档。你能在这个框架的官方网站上找到文档吗?


[Discussions][https://discuss.d2l.ai/t/1765]


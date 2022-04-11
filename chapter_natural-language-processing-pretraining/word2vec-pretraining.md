# 预训练word2vec
:label:`sec_word2vec_pretraining`

我们继续实现 :numref:`sec_word2vec`中定义的跳元语法模型。然后，我们将在PTB数据集上使用负采样预训练word2vec。首先，让我们通过调用`d2l.load_data_ptb`函数来获得该数据集的数据迭代器和词表，该函数在 :numref:`sec_word2vec_data`中进行了描述。


> PTB(Penn Tree Bank)数据集介绍：http://www.lysblog.cn:8080/blog/65
>
> 包含很多文件，但是我们只关心 data 文件夹下面的三个文件：ptb.test.txt、ptb.train.txt、ptb.valid.txt (如下图所示)。
>
> 这三个文件中的数据已经经过预处理，相邻单词之间用空格隔开，数据集中包括 9998 个不同的单词词汇，加上特殊符号 (稀有词语) 和语句结束标记符 (换行符) 在内，一共是 10000 个词汇。近年来关于语言模型方面的论文大多采用了 Mikolov 提供的这一预处理后的数据版本，由此保证论文之间具有比较性。


```python
import math
import torch
from torch import nn
from d2l import torch as d2l
```


```python
# 遇到d2l的bug，要把num_workers返回0
# http://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/word-embedding-dataset.html
d2l.get_dataloader_workers = lambda: 0
```


```python
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
# 这个download完毕后不会发消息的，不要看着老是在Downloading
```

## 跳元模型

我们通过嵌入层和批量矩阵乘法实现了跳元模型。首先，让我们回顾一下嵌入层是如何工作的。

### 嵌入层

如 :numref:`sec_seq2seq`中所述，嵌入层将词元的索引映射到其特征向量。该层的权重是一个矩阵，其行数等于字典大小（`input_dim`），列数等于每个标记的向量维数（`output_dim`）。在词嵌入模型训练之后，这个权重就是我们所需要的。


```python
# num_beddings表示，embedding_dim是嵌入的维数
# Embedding层是一个单层的MLP，本质上是**封装过的**矩阵（和它的梯度），访问矩阵需要用.weight
# 没有初始化时，里面的参数是乱的
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

    Parameter embedding_weight (torch.Size([20, 4]), dtype=torch.float32)


嵌入层的输入是词元（词）的索引。对于任何词元索引$i$，其向量表示可以从嵌入层中的权重矩阵的第$i$行获得。由于向量维度（`output_dim`）被设置为4，因此当小批量词元索引的形状为（2，3）时，嵌入层返回具有形状（2，3，4）的向量。



```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
# embed的operator()会elemwise地索引它的embedding
# 因此要求传入的矩阵是一个整型的矩阵
# 所以最后的维度是[*x.shape, embedding_size]
```




    tensor([[[-1.2000,  0.4314,  1.1021, -0.8156],
             [-2.2387, -0.7568,  0.5548,  0.1025],
             [-0.8358,  0.6318, -0.5014,  0.0929]],
    
            [[-1.7980,  0.8309, -1.3820,  0.5327],
             [ 0.0254, -1.3058,  0.4520, -0.6496],
             [ 1.7147, -1.2924,  3.0739,  0.8969]]], grad_fn=<EmbeddingBackward0>)



### 定义前向传播

在前向传播中，跳元语法模型的输入包括形状为（批量大小，1）的中心词索引`center`和形状为（批量大小，`max_len`）的上下文与噪声词索引`contexts_and_negatives`，其中`max_len`在 :numref:`subsec_word2vec-minibatch-loading`中定义。这两个变量首先通过嵌入层从词元索引转换成向量，然后它们的批量矩阵相乘（在 :numref:`subsec_batch_dot`中描述）返回形状为（批量大小，1，`max_len`）的输出。输出中的每个元素是中心词向量和上下文或噪声词向量的点积。


> 注意这里的center是索引，而不是词本身

> u.permute(dims) 表示将维度进行调换，生成一个新的Tensor
> 
> 关于矩阵乘法：https://blog.csdn.net/leo_95/article/details/89946318
> 
> - torch.mm(mat1, mat2)是最基础的矩阵乘法(matrix multiplication)，**只能实现两个二维矩阵相乘**
> - torch.bmm(mat1, mat2)的b表示batched，只能传入两个三维矩阵并且第一维必须一样，后面两维将会进行矩阵相乘
> - torch.matmal()表示张量相乘，定义有些复杂


```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    # 这里并没有生成ground_truth
    v = embed_v(center) # center.shape = [batch_size, 1], v = [batch_size, 1, embedding_size]
    u = embed_u(contexts_and_negatives) # u.shape = [batch_size, max_len, embedding_size]
    pred = torch.bmm(v, u.permute(0, 2, 1))# 
    return pred
```

让我们为一些样例输入打印此`skip_gram`函数的输出形状。



```python
# 只是为了测试形状，没有实际意义
# 传入的都是索引，每个元素都是1
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```




    torch.Size([2, 1, 4])



## 训练

在训练带负采样的跳元模型之前，我们先定义它的损失函数。

### 二元交叉熵损失

根据 :numref:`subsec_negative-sampling`中负采样损失函数的定义，我们将使用二元交叉熵损失。


> - 自定义loss也是继承nn.Module，和自定义网络模块一样
> - forward方法就是operator()
> - nn.functional是函数，不包含梯度和副作用，有时也写成`import torch.nn.functional as F`
> - 默认创建的torch.Tensor（或者torch.ones, zeros是**不带梯度的**），需要指定`requires_grad=True`

> 关于cross_entropy_loss和sigmoid的区别：https://blog.csdn.net/Just_do_myself/article/details/123393900
> !

> BCELoss
> 
> BCE指的是Binary Cross Entropy: https://zhuanlan.zhihu.com/p/89391305


```python
class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        # with_logits的意思是，传入的inputs可以直接是全连接层输出，而不是归一化的概率，内部会用sigmoid和softmax自己归一化
        # sigmoid就是个函数，返回的Tensor是带有SigmoidBackward的
        # 等价于nn.functional.binary_cross_entropy(inputs=torch.sigmoid(inputs))
        
        # 每个pred对应一个
        # 按行就是dim=0，按列就是dim=1
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        # 输出的结果是没有归一化的
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        ret = out.mean(dim=1)
        return ret

loss = SigmoidBCELoss()
```

> 一组数据：
> ```python
out is tensor([[0.2873, 0.1051, 3.3362, 0.0122],
        [1.3873, 2.3051, 0.0000, 0.0000]])
ret is tensor([0.9352, 0.9231])
> ```


回想一下我们在 :numref:`subsec_word2vec-minibatch-loading`中对掩码变量和标签变量的描述。下面计算给定变量的二进制交叉熵损失。



```python
pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2) # list的乘法，最后得到[[...], [...]]，表示两组预测结果
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]) # 两组真实分布
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]) # mask表示组内哪些有效
# 相当于除以 mask_1_count / total_count，按mask的比例将loss还原为total_count的分数
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```




    tensor([0.9352, 1.8462])



下面显示了如何使用二元交叉熵损失中的Sigmoid激活函数（以较低效率的方式）计算上述结果。我们可以将这两个输出视为两个规范化的损失，在非掩码预测上进行平均。



```python
# sigmoid将变量映射到[0, 1]，原始公式是1/(1+math.exp(-x))
# 防止概率相乘下溢，取了对数
# 由于概率值的对数是负的，取了反
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

    0.9352
    1.8462


### 初始化模型参数

我们定义了两个嵌入层，将词表中的所有单词分别作为中心词和上下文词使用。字向量维度`embed_size`被设置为100。



```python
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 定义训练阶段代码

训练阶段代码实现定义如下。由于填充的存在，损失函数的计算与以前的训练函数略有不同。



```python
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight) # 使用xavier_uniform来原地初始化embedding.weight
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

现在，我们可以使用负采样来训练跳元模型。



```python
# d2l可以根据epoch绘制loss的图形
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

    loss 0.410, 170086.4 tokens/sec on cuda:0



    
![svg](chapter_natural-language-processing-pretraining/word2vec-pretraining_files/word2vec-pretraining_31_1.svg)
    


## 应用词嵌入
:label:`subsec_apply-word-embed`

在训练word2vec模型之后，我们可以使用训练好模型中词向量的余弦相似度来从词表中找到与输入单词语义最相似的单词。



```python
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词，概率最高的肯定是自己
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

    cosine sim=0.727: intel
    cosine sim=0.691: microprocessor
    cosine sim=0.674: motorola


## 小结

* 我们可以使用嵌入层和二元交叉熵损失来训练带负采样的跳元模型。
* 词嵌入的应用包括基于词向量的余弦相似度为给定词找到语义相似的词。

## 练习

1. 使用训练好的模型，找出其他输入词在语义上相似的词。您能通过调优超参数来改进结果吗？
1. 当训练语料库很大时，在更新模型参数时，我们经常对当前小批量的*中心词*进行上下文词和噪声词的采样。换言之，同一中心词在不同的训练迭代轮数可以有不同的上下文词或噪声词。这种方法的好处是什么？尝试实现这种训练方法。





# 现代循环神经网络
:label:`chap_modern_rnn`

前一章中我们介绍了循环神经网络的基础知识，
这种网络可以更好地处理序列数据。
我们在文本数据上实现了基于循环神经网络的语言模型，
但是对于当今各种各样的序列学习问题，这些技术可能并不够用。

例如，循环神经网络在实践中一个常见问题是数值不稳定性。
尽管我们已经应用了梯度裁剪等技巧来缓解这个问题，
但是仍需要通过设计更复杂的序列模型可以进一步处理它。
具体来说，我们将引入两个广泛使用的网络，
即*门控循环单元*（gated recurrent units，GRU）和
*长短期记忆网络*（long short-term memory，LSTM）。
然后，我们将基于一个单向隐藏层来扩展循环神经网络架构。
我们将描述具有多个隐藏层的深层架构，
并讨论基于前向和后向循环计算的双向设计。
现代循环网络经常采用这种扩展。
在解释这些循环神经网络的变体时，
我们将继续考虑 :numref:`chap_rnn`中的语言建模问题。

事实上，语言建模只揭示了序列学习能力的冰山一角。
在各种序列学习问题中，如自动语音识别、文本到语音转换和机器翻译，
输入和输出都是任意长度的序列。
为了阐述如何拟合这种类型的数据，
我们将以机器翻译为例介绍基于循环神经网络的
“编码器－解码器”架构和束搜索，并用它们来生成序列。

:begin_tab:toc
 - [gru](gru.ipynb)
 - [lstm](lstm.ipynb)
 - [deep-rnn](deep-rnn.ipynb)
 - [bi-rnn](bi-rnn.ipynb)
 - [machine-translation-and-dataset](machine-translation-and-dataset.ipynb)
 - [encoder-decoder](encoder-decoder.ipynb)
 - [seq2seq](seq2seq.ipynb)
 - [beam-search](beam-search.ipynb)
:end_tab:

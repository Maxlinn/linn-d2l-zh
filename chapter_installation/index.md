# 安装
:label:`chap_installation`

我们需要配置一个环境来运行 Python、Jupyter Notebook、相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

## 安装 Miniconda

最简单的方法就是安装依赖Python 3.x的。
如果已安装conda，则可以跳过以下步骤。访问Miniconda网站，根据Python3.x版本确定适合你的系统的版本。

如果你使用macOS，假设你的Python版本是3.8（我们的测试版本），你将下载名称包含字符串“MacOSX”的bash脚本，并执行以下操作：

```bash
# 文件名可能会更改
sh Miniconda3-py38_4.10.3-MacOSX-x86_64.sh -b
```

如果你使用Linux，假设你的Python版本是3.8（我们的测试版本），你将下载名称包含字符串“Linux”的bash脚本，并执行以下操作：

```bash
# 文件名可能会更改
sh Miniconda3-py38_4.10.3-Linux-x86_64.sh -b
```

接下来，初始化终端Shell，以便我们可以直接运行`conda`。

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开当前的 shell。你应该能用下面的命令创建一个新的环境：

```bash
conda create --name d2l python=3.8 -y
```

现在激活 `d2l` 环境：

```bash
conda activate d2l
```

## 安装深度学习框架和`d2l`软件包

在安装深度学习框架之前，请先检查你的计算机上是否有可用的GPU。
例如，你可以查看计算机是否装有NVIDIA GPU并已安装。
如果你的机器没有任何GPU，没有必要担心，因为你的CPU在前几章完全够用。
但是，如果你想流畅地学习全部章节，请提早获取GPU并且安装深度学习框架的GPU版本。


你可以按如下方式安装PyTorch的CPU或GPU版本：

```bash
pip install torch==1.10.2
pip install torchvision==0.11.3
```


我们的下一步是安装`d2l`包，以方便调取本书中经常使用的函数和类：

```bash
pip install d2l==0.17.4
```

## 下载 D2L Notebook

接下来，需要下载这本书的代码。
你可以点击本书HTML页面顶部的“Jupyter 记事本”选项下载后解压代码。
或者，你可以按照如下方式进行下载：


```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```


注意：如果你没有安装`unzip`，则可以通过运行`sudo apt install unzip`进行安装。


安装完成后，你可以通过运行以下命令打开Jupyter笔记本（在Window系统的命令行窗口中运行以下命令前，需先将当前路径定位到刚下载的本书代码解压后的目录）：

```bash
jupyter notebook
```

现在，你可以在Web浏览器中打开<http://localhost:8888>（通常会自动打开）。
由此，你可以运行这本书中每个部分的代码。
在运行书籍代码、更新深度学习框架或`d2l`软件包之前，请始终执行`conda activate d2l`以激活运行时环境。
要退出环境，请运行`conda deactivate`。





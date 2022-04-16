# 优化算法
:label:`chap_optimization`

如果您在此之前按顺序阅读这本书，则已经使用了许多优化算法来训练深度学习模型。这些工具使我们能够继续更新模型参数，并使损失函数的值最小化，正如在训练集上评估的那样。事实上，任何满足于将优化视为黑盒装置，以便在简单的设置中最小化目标函数的人，都可能会知道存在着一系列此类程序的咒语（名称如“SGD”和“Adam”）。

但是，为了做得好，还需要更深入的知识。优化算法对于深度学习非常重要。一方面，训练复杂的深度学习模型可能需要数小时、几天甚至数周。优化算法的性能直接影响模型的训练效率。另一方面，了解不同优化算法的原则及其超参数的作用将使我们能够以有针对性的方式调整超参数，以提高深度学习模型的性能。

在本章中，我们深入探讨常见的深度学习优化算法。深度学习中出现的几乎所有优化问题都是*非凸*的。尽管如此，在*凸*问题背景下设计和分析算法是非常有启发性的。正是出于这个原因，本章包括了凸优化的入门，以及凸目标函数上非常简单的随机梯度下降算法的证明。

:begin_tab:toc
 - <img src="chapter_optimization/optimization-intro.ipynb" alt="optimization-intro">
 - <img src="chapter_optimization/convexity.ipynb" alt="convexity">
 - <img src="chapter_optimization/gd.ipynb" alt="gd">
 - <img src="chapter_optimization/sgd.ipynb" alt="sgd">
 - <img src="chapter_optimization/minibatch-sgd.ipynb" alt="minibatch-sgd">
 - <img src="chapter_optimization/momentum.ipynb" alt="momentum">
 - <img src="chapter_optimization/adagrad.ipynb" alt="adagrad">
 - <img src="chapter_optimization/rmsprop.ipynb" alt="rmsprop">
 - <img src="chapter_optimization/adadelta.ipynb" alt="adadelta">
 - <img src="chapter_optimization/adam.ipynb" alt="adam">
 - <img src="chapter_optimization/lr-scheduler.ipynb" alt="lr-scheduler">
:end_tab:


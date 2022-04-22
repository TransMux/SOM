<h2 align="center">✨ Self-Organized-Map (SOM)</h2>

It realizes a unified interface, with more comments and friendly novice~

Pull requests and issues are highly welcomed，Leave a star if it is helpful ! Thanks~

### Directory structure

It contains `3` individual experiments，using `main.py` you can train SOM model once which contains all visualization for a single train. 

In `demo.py` and `visualize.py`, I fired experiments on different neighbor functions and different strategies for lr decrease.(12 in total, visualize results below)

```tree
│  .gitignore
│  demo.py # Visualization of 12 experimental training model effects
│  ErrorVisualize.png
│  main.py # Single train
│  NumVisualize.png
│  README.md
│  visualize.py # Visualization of error decline curve of 12 experimental training
└─ SOM
    │  Distance.py
    │  model.py
    │  Neighbor.py # Neighbor Functions locate
    └─  __init__.py
```

### After All

从左到右每一列分别为：

* 不使用学习率下降策略
* 使用线性学习率下降策略
* 使用指数学习率下降策略，且分母为最大迭代次数
* 使用指数学习率下降策略，且分母为最大迭代次数的两倍

从上到下每一行分别为：

* 使用bubble窗函数
* 使用高斯窗函数
* 使用Triangle窗函数

![](ErrorVisualize.png)

![](NumVisualize.png)

### License

GPL

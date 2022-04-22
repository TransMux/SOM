<h2 align="center">✨ Self-Organized-Map (SOM)</h2>

It realizes a unified interface, with more comments and friendly novice~

Pull requests and issues are highly welcomed，Leave a star if it is helpful ! Thanks~

### Directory structure

It contains `3` individual experiments，using `main.py` you can train SOM model once which contains all visualization for
a single train.

In `demo.py` and `visualize.py`, I fired experiments on different neighbor functions and different strategies for lr
decrease.(12 in total, visualize results below)

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

Each column from left to right is using different Learning rate decline strategy

Each line from top to bottom is using different neighbor functions

![](ErrorVisualize.png)

![](NumVisualize.png)

### License

GPL

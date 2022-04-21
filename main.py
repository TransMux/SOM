from matplotlib import pyplot as plt

from SOM.model import SOM

from sklearn import datasets
from sklearn.preprocessing import scale

digits = datasets.load_digits(n_class=10)
data = digits.data
data = scale(data)
num = digits.target

som = SOM(
    size=(30, 30),
    feature=64,
    learning_rate=0.5,
    max_iterations=50,
)

som.fit(data)

plt.figure(figsize=(8, 8))
wmap = {}
im = 0
for x, t in zip(data, num):
    w = som.get_winner(x)
    wmap[w] = im
    plt.text(w[0] + .5, w[1] + .5, str(t), color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold', 'size': 11})
    im = im + 1
plt.axis([0, som.size[0], 0, som.size[1]])
plt.show()

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
    learning_rate=10.5,
    max_iterations=10,
)

som.fit(data)

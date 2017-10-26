"""
Created by Alex Wang
On 2017-10-25
test cnn on mnist dataset
"""
import tensorlayer as tl

def train_mnist(FLAGS):
    print("start_train_mnist")
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,28,28),path=FLAGS.input_path)

def test_mnist(FLAGS):
    print("start_test_mnist")
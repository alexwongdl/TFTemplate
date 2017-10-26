import tensorlayer as tl


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,28,28),path='../data/MNIST')
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(X_test[0][:][:])
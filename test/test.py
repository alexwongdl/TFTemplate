import tensorlayer as tl
import nltk
import numpy as np
# nltk.download('punkt')

def file_space(token_list):
    new_list = []
    for token in token_list:
        if token.strip() != "":
            new_list.append(token)
    return new_list

def test_nltk():
    sentence = "Durant l'Antiquité, l'astrométrie, qui    est la mesure de la position des étoiles et des planètes, est la principale occupation des astronomes.".lower()
    tokens = nltk.word_tokenize(sentence)
    print((tokens))
    print(file_space(tokens))

def test_mnist():
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,28,28),path='../data/MNIST')
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    # print(X_test[0])
    # print(y_test.shape)
    # print(y_test.ndim)
    # print(y_test[1:10])

def test_numpy():
    a = np.round(3.4)
    print(a * 3)

if __name__ == '__main__':
    # test_mnist()
    # test_nltk()
    test_numpy()



# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs

# Import SVM models
from HMISVM import HMISVM


if __name__ == '__main__':
    # Make linearly separable dataset
    dataset_size = 1000
    X, y = make_blobs(n_samples=dataset_size, centers=2, cluster_std=0.5, random_state=4)
    y = (2 * y - 1)

    # Split dataset
    split = int(0.9 * dataset_size)
    train_X, train_y = X[:split], y[:split]
    test_X, test_y = X[split:], y[split:]

    # Train SVM
    svm = HMISVM()
    svm.fit(train_X, train_y)

    # Predict and calculate accuracy
    pred_y = svm.predict(test_X)

    correct = 0
    for i in range(len(test_y)):
        if int(pred_y[i]) == int(test_y[i]):
            correct += 1
    print(f'Accuracy : {correct/len(test_y)}')

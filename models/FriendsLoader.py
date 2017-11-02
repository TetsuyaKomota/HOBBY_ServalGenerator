# coding = utf-8

import glob
import cv2
import numpy as np

def load_data(test_rate=0):
    X = {}
    X["nokemonohainai"] = []
    dataPaths = glob.glob("tmp/friends/*")
    for dpath in dataPaths:
        X["nokemonohainai"].append(cv2.imread(dpath))

    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    for i, label in enumerate(X):
        testSize = int(len(X[label])*test_rate)
        X_train = X_train + X[label][testSize:]
        X_test  = X_test  + X[label][:testSize]
        y_train = y_train + [i for _ in range(len(X[label])-testSize)]
        y_test  = y_test  + [i for _ in range(testSize)]

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))

if __name__ == "__main__":
    (X_train, y_train), (_, _) = load_data()
    print(X_train.shape)
    print(X_train[0])

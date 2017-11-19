# coding = utf-8

import glob
import os
import cv2
import matplotlib.pyplot as plt

impaths  = glob.glob("tmp/friends/*.png")
sizeList = []

for impath in impaths:
    img = cv2.imread(impath)
    sizeList.append(img.shape[0])

plt.hist(sizeList, bins = 50)
plt.show()

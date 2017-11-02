# coding = utf-8

import cv2
import glob
import os

# http://peaceandhilightandpython.hatenablog.com/entry/2016/01/08/000857

gs = []
gs.append(glob.glob("tmp/dataset/fromMovies/*"))
gs.append(glob.glob("tmp/dataset/noized/*"))
gs.append(glob.glob("tmp/dataset/duplicated/*"))

for g in gs:
    for f in g:
        img = cv2.imread(f)
        cv2.imwrite(f[:-4]+"_rev"+f[-4:], cv2.flip(img, 1))

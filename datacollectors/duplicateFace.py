# coding = utf-8

import cv2
import glob
import os
import numpy as np

files = glob.glob("tmp/dataset/fromMovies/*")

def kakeru(lis):
    output = 1
    for l in lis:
        output *= l
    return output

for f in files:
    img    = cv2.imread(f)
    noize  = np.random.rand(kakeru([d for d in img.shape])).reshape(img.shape)
    noize -= 0.5
    noize *= 10
    fname  = "noized/n_"
    img    = np.array(img, dtype = "float64")
    img   += noize # noized
    img    = np.clip(img, 0, 255)
    img    = np.array(img, dtype="uint8")
    # cv2.imshow("noized img", img)
    # cv2.waitKey(0)
    cv2.imwrite("tmp/dataset/"+fname + os.path.basename(f), img)
    """
    noize     = np.array(img, dtype = "float64")
    d         = 1
    img[:-d]  = noize[d:]
    img[-d:]  = noize[:d]
    fname  = "duplicated/d_"

    cv2.imwrite("tmp/dataset/"+fname + os.path.basename(f), img)
    """

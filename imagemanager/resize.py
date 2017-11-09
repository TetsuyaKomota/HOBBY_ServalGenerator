# coding = utf-8

import cv2
import glob
import os

fpaths = glob.glob("tmp/output/*")
resizeTo = 2000

xmax = 0
ymax = 0
xmin = 10000
ymin = 10000
for fpath in fpaths:
    
    if os.path.isdir(fpath) == True:
        print("Directory:" + fpath)
        continue
    
    img = cv2.imread(fpath)
    
    size = img.shape[:2]

    ymax = max(ymax, size[0])
    xmax = max(xmax, size[1])
    ymin = min(ymin, size[0])
    xmin = min(xmin, size[1])

    resized = cv2.resize(img, (resizeTo, resizeTo), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite("tmp/output/resized/"+os.path.basename(fpath), resized)


print("xmax:" + str(xmax))
print("ymax:" + str(ymax))
print("xmin:" + str(xmin))
print("ymin:" + str(ymin))

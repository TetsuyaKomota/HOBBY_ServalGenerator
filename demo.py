# -*- coding: utf-8 -*-

from keras.models import model_from_json
from keras.optimizers import Adam

import cv2
import os
import numpy as np

from setting import DEMO_EPOCH
from setting import G_LR
from setting import G_BETA

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
    
    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent
    
    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
            
# Generator クラス
class Generator:
    def __init__(self, epoch):
        g_json_path = "tmp/save_models/generator.json"
        if os.path.exists(g_json_path):
            with open(g_json_path, "r", encoding="utf-8") as f:
               self.generator = model_from_json(f.read())
        else:
            print("could not found json file")
            exit()
        g_weights_load_path = "tmp/save_models/g_w_" + str(epoch) + ".h5"
        if os.path.exists(g_weights_load_path):
            self.generator.load_weights(g_weights_load_path, by_name=False)
        else:
            print("could not found weights of epoch:" + str(epoch))
            exit()
        self.generator.trainable = False
        g_opt = Adam(lr=G_LR, beta_1=G_BETA)
        self.generator.compile(loss="binary_crossentropy", optimizer=g_opt)

    def pick(self, n):
        img = self.generator.predict(n, verbose=0)[0]
        img = img * 127.5 + 127.5
        img = img.astype(np.uint8)
        print(img)
        img = cv2.resize(img, (160, 160), interpolation = cv2.INTER_LINEAR)
        cv2.imshow("G", img)

if __name__ == "__main__":
    #入力画像
    read = cv2.imread("tmp/demo/field.png")
   
    #表示するWindow名
    window_name = "input window"
   
    # generator のロード
    generator = Generator(DEMO_EPOCH)
 
    #画像の表示
    cv2.imshow(window_name, read)
    
    #コールバックの設定
    mouseData = mouseParam(window_name)
    
    while 1:
        cv2.waitKey(20)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_MOUSEMOVE:
            pos = mouseData.getPos()
            pos = [(pos[r%2]/750)*2 - 1 for r in range(100)]
            # print(pos)
            generator.pick(np.array([pos]))
            # cv2.waitKey(0)
       #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break;
            
    cv2.destroyAllWindows()            
    print("Finished")

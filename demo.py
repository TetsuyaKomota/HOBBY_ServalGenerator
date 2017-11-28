# -*- coding: utf-8 -*-

from keras.models import model_from_json
from keras.optimizers import Adam

import cv2
import os
import numpy as np

from setting import DEMO_EPOCH
from setting import G_LR
from setting import G_BETA
from setting import E_LR
from setting import E_BETA

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
        # print(img)
        img = cv2.resize(img, (160, 160), interpolation = cv2.INTER_LINEAR)
        cv2.imshow("G", img)

# Encoder クラス
class Encoder:
    def __init__(self, epoch):
        e_json_path = "tmp/save_models/encoder.json"
        if os.path.exists(e_json_path):
            with open(e_json_path, "r", encoding="utf-8") as f:
               self.encoder = model_from_json(f.read())
        else:
            print("could not found json file")
            exit()
        e_weights_load_path = "tmp/save_models/e_w_" + str(epoch) + ".h5"
        if os.path.exists(e_weights_load_path):
            self.encoder.load_weights(e_weights_load_path, by_name=False)
        else:
            print("could not found weights of epoch:" + str(epoch))
            exit()
        self.encoder.trainable = False
        e_opt = Adam(lr=E_LR, beta_1=E_BETA)
        self.encoder.compile(loss="binary_crossentropy", optimizer=e_opt)

    def pick(self, imgName):
        img = cv2.imread("tmp/friends/"+imgName)
        img = (img.astype(np.float32)-127.5)/127.5
        return self.encoder.predict(np.array([img]), verbose=0)[0]


# 四隅の値と座標値を引数に，入力空間上の座標を取得する
# 四隅の値は numpy.array 前提
def mapping(cornerList, pos, fieldSize):
    # 座標を [0, 1] 正規化する
    normalPos = [pos[0]/fieldSize, pos[1]/fieldSize]
    print(normalPos)
    # x方向の2辺の内点を取る
    left  = cornerList[0]*normalPos[0] + cornerList[1]*(1-normalPos[0])
    right = cornerList[2]*normalPos[0] + cornerList[3]*(1-normalPos[0])
    # 内点同士の内点を取る
    return left*normalPos[1] + right*(1-normalPos[1])

if __name__ == "__main__":
    #入力画像
    read = cv2.imread("tmp/demo/field.png")
    img  = np.zeros_like(read)
    size = read.shape[0]
 
    #表示するWindow名
    window_name = "input window"
   
    # generator のロード
    generator = Generator(DEMO_EPOCH)

    # Encoder のロード
    encoder   = Encoder(DEMO_EPOCH)
 
    #画像の表示
    cv2.imshow(window_name, read)
    
    #コールバックの設定
    mouseData = mouseParam(window_name)
   
    # 入力空間の四隅のノイズ値
    cornerList = []
    cornerList.append(encoder.pick("01_01176_00.png")) # サーバルちゃん
    cornerList.append(encoder.pick("08_03384_01.png")) # かばんちゃん
    cornerList.append(encoder.pick("11_13104_00.png")) # アライさん
    cornerList.append(encoder.pick("11_15000_00.png")) # フェネック
    """
    cornerList.append(encoder.pick("01_04440_00.png")) # サーバルちゃん
    cornerList.append(encoder.pick("01_05856_00.png")) # サーバルちゃん
    cornerList.append(encoder.pick("01_23496_00.png")) # サーバルちゃん
    cornerList.append(encoder.pick("06_14112_00.png")) # サーバルちゃん
    """
    while 1:
        cv2.waitKey(20)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_MOUSEMOVE:
            pos = mouseData.getPos()
            # pos = [(pos[r%2]/750)*2 - 1 for r in range(100)]
            pos = mapping(cornerList, pos, size)
            generator.pick(np.array([pos]))
       #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break;
            
    cv2.destroyAllWindows()            
    print("Finished")

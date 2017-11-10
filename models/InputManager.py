# coding = utf-8

# Generator に渡すノイズの選び方を決定する

# ルール
# ・ノイズセットは丁度4つ (描画の都合)
# ・エポック数とDの評価値リストを引数にノイズセットを返す関数
# ・最初の処理(エポック0)は dList = None でたたかれるのでその時の処理

import dill
import glob
import numpy as np
import os

from setting import NOIZE_SIZE
from setting import BATCH_SIZE
SAVE_NOIZE_PATH = "tmp/save_noizes/"
# ノイズクラス

class InputManager:
    def __init__(self, methodIdx):
        self.noizeList = []
        for i in range(4):
            dillName = "forLearn_" + str(i) + ".dill"
            self.noizeList.append(self.loadorGenerateNoizeSet(dillName))
        if   methodIdx == 2:
            self.next = self.next2
        elif methodIdx == 1:
            self.next = self.next1
        else:
            self.next = self.next0
            
    # 長さ dim の 乱数配列を num 個持つ2次元配列を返す
    # 主に generator に渡す入力を作るのに使用する
    def makeNoize(self, dim, num):
        return np.array([np.random.uniform(-1,1,dim) for _ in range(num)])

    # ノイズのセットをロード，ない場合は生成してセーブする
    def loadorGenerateNoizeSet(self, dillName):
        n_sample_path = SAVE_NOIZE_PATH + dillName

        if os.path.exists(n_sample_path):
            with open(n_sample_path, "rb") as f:
                n_sample = dill.load(f)
        else:
            n_sample = self.makeNoize(NOIZE_SIZE, BATCH_SIZE)
            with open(n_sample_path, "wb") as f:
                dill.dump(n_sample, f)

        return n_sample

    # =====================================================
    # ノイズセットの要求メソッド
    # 0 : 順繰りに渡す
    def next0(self, epoch, dList):
        return self.noizeList[epoch%(len(self.noizeList))]

    # 1 : 100 エポックごとに 1, 2, 3 をサイクルし，
    #     3 の時は評価の低いノイズを更新する
    def next1(self, epoch, dList):
        if dList is None:
            return self.noizeList[1]
        l = len(self.noizeList)
        nextIdx = int(epoch/100)%(l-1)+1
        if nextIdx == 0: 
            resultList = [(i, r[0]) for i, r in enumerate(dList)]
            resultList = sorted(resultList, key=(lambda t:t[1]))[:10]
            for r in resultList:
                newrand = np.random.uniform(-1, 1, NOIZE_SIZE)
                self.noizeList[l-1][r[0]] = newrand
            dillName = "forLearn_" + str(l-1) + ".dill"
            with open(SAVE_NOIZE_PATH+dillName,"wb") as f:
                dill.dump(self.noizeList(l-1), f)
        return self.noizeList[nextIdx]

    # 2 : 1 エポックごとに 0, 1, 2, 3 をサイクルし，
    #     すべてのセットで評価の低いノイズを更新する
    def next2(self, epoch, dList):
        if dList is None:
            return self.noizeList[1]
        l = len(self.noizeList)
        nextIdx = epoch%l
        resultList = [(i, r[0]) for i, r in enumerate(dList)]
        resultList = sorted(resultList, key=(lambda t:t[1]))[:45]
        for r in resultList:
            newrand = np.random.uniform(-1, 1, NOIZE_SIZE)
            self.noizeList[(nextIdx-1)%l][r[0]] = newrand
        dillName = "forLearn_" + str((nextIdx-1)%l) + ".dill"
        with open(SAVE_NOIZE_PATH+dillName,"wb") as f:
            dill.dump(self.noizeList[(nextIdx-1)%l], f)
        return self.noizeList[nextIdx]


    # =====================================================

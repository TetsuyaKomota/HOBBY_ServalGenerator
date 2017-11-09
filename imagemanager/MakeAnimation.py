import glob
import os
from PIL import Image
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
if __name__ == "__main__":
     
    folderName = "tmp/output"


    #画像ファイルの一覧を取得
    picList = glob.glob(folderName + "\*.png")
     
    #figオブジェクトを作る
    fig = plt.figure()
     
    #空のリストを作る
    ims = []
     
    #画像ファイルを順々に読み込んでいく
    for i in range(len(picList)):
         
        #1枚1枚のグラフを描き、appendしていく
        tmp = Image.open(picList[i])
        ims.append(plt.imshow(tmp))     
     
    #アニメーション作成    
    if os.path.exists("tmp/imagemanager") == False:
        os.mkdir(folderName) 
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
    ani.save("tmp/imagemanager/MakeAnimation_result.gif")

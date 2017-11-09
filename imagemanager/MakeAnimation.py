import glob
import os
from PIL import Image
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 実行するには imagemagick が必要
# インストーラ : https://www.imagemagick.org/script/download.php
# パス等設定   : https://endoyuta.com/2017/02/08/python3-matplotlib%E3%81%A7%E3%82%A2%E3%83%8B%E3%83%A1%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3gif%E3%82%92%E3%81%A4%E3%81%8F%E3%82%8B/


 
if __name__ == "__main__":
    
    print("[MakeAnimation]start")
 
    folderName = "tmp/output"


    #画像ファイルの一覧を取得
    picList = glob.glob(folderName + "\*.png")
     
    #figオブジェクトを作る
    fig = plt.figure()

    # 軸を消す
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    
    #空のリストを作る
    ims = []
    
    size = len(picList)
    var  = "----------"
 
    print("[MakeAnimation]input image:" + str(size))
    
    #画像ファイルを順々に読み込んでいく
    for i in range(size):
        
        #1枚1枚のグラフを描き、appendしていく
        tmp = Image.open(picList[i])
        ims.append([plt.imshow(tmp)])     
    
        # 進行を表示
        if i%(int(size/9)) == 0:
            var = "+" + var[:-1]
            print("[MakeAnimation]" + var)
 
    var = "+" + var[:-1]
    print("[MakeAnimation]" + var)

    #アニメーション作成    
    if os.path.exists("tmp/imagemanager") == False:
        os.mkdir("tmp/imagemanager") 
    print("[MakeAnimation]save gif")
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
    ani.save("tmp/imagemanager/MakeAnimation_result.gif", writer="imagemagick")

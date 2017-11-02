# coding = utf-8

import cv2 


cascade_path = "../rizeaya/tmp/cascades/lbpcascade_animeface.xml"

def main(movieName):

    # 動画の読み込み
    cap = cv2.VideoCapture("tmp/movies/" + movieName + ".mp4")
    
    #カスケード分類器の特徴量を取得
    cascade = cv2.CascadeClassifier(cascade_path)

    # F フレームごとに画像を取得する
    F = 24
        
    count = 0

    # 動画終了まで繰り返し
    while(cap.isOpened()):
       
        # フレームを取得
        ret, frame = cap.read()
        
        count += 1
        if count % F != 0:
            continue

        try:
            #グレースケール変換
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #物体認識（顔認識）の実行
            facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(64, 64))
        except:
            print("invalid input")
            break
        i = 0;
        for rect in facerect:
            #顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            pad = 0.1
            width = rect[2]
            height = rect[3]
            dst = frame[y:y+height, x:x+width]
            new_image_path = 'tmp/dataset/fromMovies/' + movieName + "_" "{0:05d}".format(count) + "_" + "{0:02d}".format(i) + ".png"
            cv2.imwrite(new_image_path, dst)
            i += 1
        # 保存
        # cv2.imwrite("tmp/fromMovies/" + "{0:05d}".format(count) + ".png", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(10, 14):
        main("{0:02d}".format(i))

# Encoder がうまく働いているかのテスト

import glob
import cv2
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam

from setting import G_LR
from setting import G_BETA

model_path = "tmp/save_models/"
epoch      = 100

# モデルのロード
with open(model_path+"generator.json", "r", encoding="utf-8") as f:
    generator   = model_from_json(f.read())
    weight_path = model_path + "g_w_" + str(epoch) + ".h5"
    generator.load_weights(weight_path, by_name=False)
with open(model_path+"encoder.json", "r", encoding="utf-8") as f:
    encoder     = model_from_json(f.read())
    weight_path = model_path + "e_w_" + str(epoch) + ".h5"
    encoder.load_weights(weight_path, by_name=False)

model = Sequential([encoder, generator])
model.trainable = False
g_opt = Adam(lr=G_LR, beta_1=G_BETA)
model.compile(loss="mean_squared_error", optimizer=g_opt)


imgpaths = glob.glob("tmp/friends/*.png")

for imgpath in imgpaths:
    real = cv2.imread(imgpath)
    real = real.astype(np.float32)
    real = (real-127.5)/127.5
    fake = model.predict(np.array([real]), verbose=0)[0]
    fake = fake*127.5 + 127.5
    fake = fake.astype(np.uint8)
    cv2.imshow("real", cv2.resize(real, (160, 160), interpolation=cv2.INTER_LINEAR))
    cv2.imshow("fake", cv2.resize(fake, (160, 160), interpolation=cv2.INTER_LINEAR))
    key = cv2.waitKey(0)
    if key == ord("a"):
        break

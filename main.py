# coding = utf-8

# from models import gan
from models import AWSgan as gan
from setting import IS_RESET_MODEL
from setting import IS_RESET_NOIZE

import os

if __name__ == "__main__":
    # IS_RESET_MODEL が True なら， save_model
    # IS_RESET_NOIZE が True なら， save_noize
    # を削除する
    if IS_RESET_MODEL == True:
        os.remove("tmp/save_models/discriminator.json")
        os.remove("tmp/save_models/discriminator.h5")
        os.remove("tmp/save_models/generator.json")
        os.remove("tmp/save_models/generator.h5")
    if IS_RESET_NOIZE == True:
        os.remove("tmp/save_noizes/forLearn_0.dill")
        os.remove("tmp/save_noizes/forLearn_1.dill")
        os.remove("tmp/save_noizes/forLearn_2.dill")
        os.remove("tmp/save_noizes/forLearn_3.dill")

    gan.train()

# coding = utf-8

# from models import gan
# from models import AWSgan as gan
# from models import EncoderLearning as gan
# from models import EncodedDCGAN as gan
# from models import EDCGANv5 as gan
# from models import HyperAutoEncoder as gan
from models import ProgressiveGrowedGAN as gan
from setting import IS_RESET_MODEL
from setting import IS_RESET_NOIZE

import os
import glob

if __name__ == "__main__":
    # IS_RESET_MODEL が True なら， save_model
    # IS_RESET_NOIZE が True なら， save_noize
    # を削除する
    if IS_RESET_MODEL == True:
        files = glob.glob("tmp/save_models/*.*")
        for f in files:
            os.remove(f)
    if IS_RESET_NOIZE == True:
        files = glob.glob("tmp/save_noizes/*.*")
        for f in files:
            os.remove(f)

    gan.train()

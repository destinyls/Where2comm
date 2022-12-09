import os
import numpy as np

npy_pred = "/root/1482_pred.npy"

if __name__ == "__main__":
    pred_box3d = np.load(npy_pred)
    print("pred_box3d: ", pred_box3d.shape)

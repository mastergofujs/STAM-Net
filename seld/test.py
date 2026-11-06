import h5py
import numpy as np
import pandas as pd
import torch
import csv
import os

def main():
    # 原始文件名
    # "/data1/DCASE2021/_hdf5/DCASE2021/meta/dev/"
    # "/data1/DCASE2021/_hdf5/DCASE2021/data/24000fs/dev/foa/"
    folder_path = "/data1/DCASE2021/_hdf5/DCASE2021/meta/dev/"
    for i in range(6):
        # fold1-fold6
        for j in range(2):
            for k in range(50):
                # old
                if k+1 < 10:
                    old_file_path = os.path.join(folder_path, f"fold{i+1}_room{j+1}_mix00{k+1}.h5")
                elif k+1 >= 10:
                    old_file_path = os.path.join(folder_path, f"fold{i+1}_room{j+1}_mix0{k+1}.h5")

                # new
                if k+1 < 10:
                    new_file_path = os.path.join(folder_path, f"fold{i+1}_room{j+1}_mix00{k+1}_ov1.h5")
                elif k+1 <= 25:
                    new_file_path = os.path.join(folder_path, f"fold{i+1}_room{j+1}_mix0{k+1}_ov1.h5")
                elif k+1 >25:
                    new_file_path = os.path.join(folder_path, f"fold{i+1}_room{j+1}_mix0{k+1}_ov2.h5")

                os.rename(old_file_path, new_file_path)
                print(" {}  →→→→→→→→→  {}".format(old_file_path, new_file_path))
    


if __name__ == "__main__":
    main()
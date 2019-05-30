# Author: QChen
# Date: 2019.05.29
import os
import shutil

dataDir = "./Data"
files = os.listdir("./jpg")

for file in files:
    index = (((file.split('.'))[0]).split('_'))[1]
    C = int(index) // 81
    desDir = os.path.join(dataDir, str(C))
    if not os.path.exists(desDir):
        os.makedirs(desDir)
        shutil.move("./jpg/" + file, desDir)
    else:
        shutil.move("./jpg/" + file, desDir)

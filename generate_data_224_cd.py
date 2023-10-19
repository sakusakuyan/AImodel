# モジュールのインポート
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection


# パラメーターの初期化
classes = ["cat", "dog"]
num_classes = len(classes)
image_size = 224

# 画像の読み込みとNumPy配列への変換
X = [] #リスト
Y = [] #リスト

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image) 
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
# サンプルの学習と検証の比率を変更
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.10, random_state=42)
xy = (X_train, X_test, y_train, y_test)
np.save("./imagefiles_224_cd.npy", xy)
#np.save("./imagefiles_224_cd.npy", np.array(xy, dtype=object))
#np.save("./cd_imagefiles.npy", xy)
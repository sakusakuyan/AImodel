import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model,load_model
from PIL import Image
import sys

# パラメーターの初期化
classes = ["cat", "dog"]
num_classes = len(classes)
image_size = 224

# 引数から画像ファイルを参照して読み込む
image = Image.open(sys.argv[1])
image = image.convert("RGB")
image = image.resize((image_size,image_size))
data = np.asarray(image) / 255.0
X = []
X.append(data)
X = np.array(X)

# モデルのロード
#model = load_model('./vgg16_transfer_cd.h5')
model = load_model('./vgg16_cd_transfer.h5')
#model = load_model('./vgg16_transfer_cdm.h5')
result = model.predict([X])[0]

#推論結果の大きい方を表示する
#predicted = result.argmax()
#percentage = int(result[predicted] * 100)
#print(classes[predicted], percentage)

# 推論結果を表示するためのループ
for i, score in enumerate(result):
    percentage = round(score * 100, 2) #少数第二位まで表示
    print(f"{classes[i]}: {percentage}%")


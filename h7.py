import cv2
import numpy as np
import argparse
import random
import time
import serial

# シリアル通信の設定
ser = serial.Serial('/dev/ttyUSB0', 115200)

# 人工知能モデルへ入力する画像の調整パラメータ
IN_WIDTH = 300
IN_HEIGHT = 300

# Mobilenet SSD COCO学習済モデルのラベル一覧の定義
CLASS_LABELS = {0: 'background',
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
                10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
                38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
                65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
                73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
                81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                90: 'toothbrush'}

# 検出したいクラスのID定義
person_class_id = 1
bicycle_class_id = 2
car_class_id = 3
motorcycle_class_id = 4

# 引数（コマンドラインのオプション指定）の定義
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--pbtxt', required=True,
                help='path to pbtxt file')
ap.add_argument('-w', '--weights', required=True,
                help='path to TensorFlow inference graph')
ap.add_argument('-c', '--confidence', type=float, default=0.3,
                help='minimum probability')
args = vars(ap.parse_args())

colors = {}
# ラベル毎の枠色をランダムにセット
random.seed()
for key in CLASS_LABELS.keys():
    colors[key] = (random.randrange(255),
                   random.randrange(255),
                   random.randrange(255))

# 人工知能モデルの読み込み
try:
    print('モデル読み込み...')
    net = cv2.dnn.readNet(args['weights'], args['pbtxt'])
except cv2.error as e:
    print(f'Error while loading the model: {e}')
    ser.close()
    exit(1)

# ビデオカメラ開始
try:
    print('ビデオカメラ開始...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video device.")
except cv2.error as e:
    print(f'Error while starting the camera: {e}')
    ser.close()
    exit(1)

# OpenCVのチックメータ（ストップウォッチ）機能をtmという名前で使えるようにする
tm = cv2.TickMeter()

# 前のフレームの検出結果を保存する変数を初期化
previous_object_detected = False
# カメラを起動したときに検知 OFF 状態とする
object_detected = False

try:
    while True:
        # カメラからの画像を読み込む
        ret, frame = cap.read()
        if not ret:
            break

        # 高さと幅情報を画像フレームから取り出す
        (frame_height, frame_width) = frame.shape[:2]

        # 画像フレームを調整しblob形式へ変換
        blob = cv2.dnn.blobFromImage(frame, size=(IN_WIDTH, IN_HEIGHT), swapRB=False, crop=False)
                
        # blob形式の入力画像を人工知能にセット
        net.setInput(blob)
        
                # 画像を人工知能へ流す
        tm.reset()
        tm.start()
        detections = net.forward()
        tm.stop()
        
        # 検出された物体の種別フラグを初期化
        person_detected = False
        bicycle_detected = False
        car_detected = False
        motorcycle_detected = False
        
        # 検出数（mobilenet SSDでは100）を繰り返す
        for i in range(detections.shape[2]):
            # i番目の検出オブジェクトの正答率を取り出す
            confidence = detections[0, 0, i, 2]

            # 正答率がしきい値を下回ったら何もしない
            if confidence < args['confidence']:
                continue

            # 検出物体の種別と座標を取得
            class_id = int(detections[0, 0, i, 1])

            if class_id == person_class_id:
                person_detected = True
            elif class_id == bicycle_class_id:
                bicycle_detected = True
            elif class_id == car_class_id:
                car_detected = True
            elif class_id == motorcycle_class_id:
                motorcycle_detected = True

            # 枠をフレームに描画
            start_x = int(detections[0, 0, i, 3] * frame_width)
            start_y = int(detections[0, 0, i, 4] * frame_height)
            end_x = int(detections[0, 0, i, 5] * frame_width)
            end_y = int(detections[0, 0, i, 6] * frame_height)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[class_id], 2)

            # 物体の種別を示す person といったラベルと確信度を label にセット
            label = CLASS_LABELS[class_id]
            label += ': ' + str(round(confidence * 100, 2)) + '%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (start_x, start_y - label_size[1]), (start_x + label_size[0], start_y + base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        ai_time = tm.getTimeMilli()
        cv2.putText(frame, '{:.2f} ms'.format(ai_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

        # 検知が OFF から ON に切り替わった場合に"A"を送信
        if not object_detected and (person_detected or bicycle_detected or car_detected or motorcycle_detected):
            ser.write(b'A')
            object_detected = True
            print("DETECTED")
        # 検知が ON から OFF に切り替わった場合に"UNDETECTED"と表示
        elif object_detected and not (person_detected or bicycle_detected or car_detected or motorcycle_detected):
            #ser.write(b'B')
            object_detected = False
            print("UNDETECTED")
            
        # フレームを画面に描画
        cv2.imshow('Live', frame)

        if cv2.waitKey(1) >= 0:
            break

except KeyboardInterrupt:
    pass

# 終了処理
ser.close()
cap.release()
cv2.destroyAllWindows()
print("blink end")


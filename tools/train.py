# /// script
# dependencies = [
#   "ultralytics",
# ]
# ///
"""copied from https://qiita.com/hirekatsu0523/items/f2f0e1a0f8a9ea92d913"""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # yolov8n/s/m/l/xのいずれかを指定。多クラスの検出であるほど大きいパラメータが必要
model.train(data="./data.yml", epochs=300, batch=20)  # 先ほど作成したデータセット内のyamlファイルまでのパスを指定

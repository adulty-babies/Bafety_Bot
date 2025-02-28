# /// script
# dependencies = [
#   "opencv-python",
#   "requests",
#   "ultralytics",
#    "lap",
# ]
# ///
import sys
from os import getenv

import cv2
import requests
from ultralytics import YOLO

LABELS = (
    "1円玉",
    "5円玉",
    "10円玉",
    "50円玉",
    "500円玉",
    "乾電池",
    "タバコ",
    "ペットボトルのキャップ",
    "100円玉",
    "薬",
)

webhook_url: str
if (url := getenv("WEBHOOK_URL")) is not None:
    webhook_url = url
else:
    raise ValueError("WEBHOOK_URL is not set.")


def generate_message(c: str):
    return {
        "content": f"赤ちゃんの周りに誤飲しやすいものを検知しました！\n 今すぐ確認してください！！\n{c}",
        "username": "Bafety_Bot",
    }


def realtime_object_detection(fp: str):
    model = YOLO(fp)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    consecutive_detection_count = 0
    cooldown_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break

            results = model.track(frame)
            detection_messages: list[str] = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detection_messages.append(
                        f"座標: {x1, y1} から {x2, y2},\nラベル: {LABELS[int(box.cls[0])]},\n 信頼度: {box.conf[0]:.2%}"
                    )
            detection_info = "\n".join(detection_messages)

            if detection_info:
                print("検出結果：\n", detection_info)
                consecutive_detection_count += 1
            else:
                print("未検出")
                consecutive_detection_count = 0

            if cooldown_counter > 0:
                cooldown_counter -= 1

            if consecutive_detection_count >= 10 and cooldown_counter == 0:
                try:
                    response = requests.post(webhook_url, json=generate_message(detection_info))
                    if response.status_code == 204:
                        print("メッセージが正常に送信されました！")
                    else:
                        print(f"エラーが発生しました。 Status: {response.status_code}, Response: {response.text}")
                except Exception as e:
                    print(f"リクエスト中に例外が発生しました: {e}")

                consecutive_detection_count = 0
                cooldown_counter = 60
    finally:
        cap.release()


def main():
    length = len(sys.argv)
    fp = "./weights/best.pt"
    if length == 2:
        fp = sys.argv[1]
    elif length > 2:
        print("Usage: python script.py [model_path]")
        sys.exit(1)

    realtime_object_detection(fp)


if __name__ == "__main__":
    main()

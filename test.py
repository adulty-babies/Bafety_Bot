# /// script
# dependencies = [
#   "opencv-python",
#   "ultralytics",
#    "lap",
# ]
# ///
import sys

import cv2
from ultralytics import YOLO


def run(video_path: str, output_path: str, model_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    annotated_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()
        else:
            break
        annotated_frames.append(annotated_frame)

    height, width, _ = annotated_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame in annotated_frames:
        video.write(frame)
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    print("finished")


def main():
    length = len(sys.argv)
    if length != 3 and length != 4:
        print("Usage: python script.py <video_path> <output_path> [model_path]")
        sys.exit(1)

    _, video_path, output_path = sys.argv[:3]

    model_path = "./weights/best.pt"
    if len(sys.argv) == 4:
        model_path = sys.argv[3]

    run(video_path, output_path, model_path)


if __name__ == "__main__":
    main()

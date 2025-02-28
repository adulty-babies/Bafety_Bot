"""pre process script for training at GPU server.
this script works for...
- generate labels for YOLO from label-studio style json-min file.
- generate uri mapping for downloading images from s3.
- generate template for data.yml.
- copy necessary files for training.
- compress necessary files for uploading to GPU server.

if you done annotation at label-studio, you should export json-min from label-studio.
and run this script with the json file path.
you can get learn.zip for ALL NECESSARY FILES for training at GPU server.
so, you can upload learn.zip to GPU server, unzip it, `chmod +x run.sh`, install necessary packages, and start training by `./run.sh`.
you can find the trained model at `./runs/train/weights/best.pt` or `./runs/train[0-9]+/weights/best.pt` (or any other *.pt files).

Directory structure of files required for preprocess.py
```
learn/               # directory for GPU server.
> datasets/          # data for training.
> > train/           # 80% of data. for training. labels/*.txt and images/*.png are paired (by name).
> > > images/        # images for training.
> > > labels/        # labels for training.
> > valid/           # 20% of data. for validation. labels/*.txt and images/*.png are paired (by name).
> > > images/        # images for validation.
> > > labels/        # labels for validation.
> data_template.yml  # template for data.yml. __file__ will be replaced before training by postprocess.py.
> download_path.json # download info for postprocess.py.
> env.tml            # save s3 credentials.
> postprocess.py     # postprocess script for training at GPU server. called by run.sh.
> run.sh             # script for starting training at GPU server.
> train.py           # training script for GPU server. called by run.sh.
learn.zip            # zip file for uploading to GPU server. compressed learn/ directory.
```
"""

import json
import os
import shutil
import zipfile
from sys import argv
from typing import Any, NamedTuple

DATA_YAML_TEMPLATE = """
train: ./train/images/
val: ./valid/images/
path: __file__

nc: {}
names:
{}
"""

type AnnotationInfo = tuple[int, float, float, float, float]


class AnnotationData(NamedTuple):
    annotations: list[AnnotationInfo]
    s3_path: str  # s3 uri


def gen_yolo_data(data: dict[Any, Any], labels: list[str], rename: dict[str, str]) -> AnnotationData:
    annotations: list[AnnotationInfo] = []
    for label in data["label"]:
        x: float = (label["x"] + label["width"] / 2) / 100  # center of x direction (percent)
        y: float = (label["y"] + label["height"] / 2) / 100  # center of y direction (percent)
        w: float = label["width"] / 100  # width (percent)
        h: float = label["height"] / 100  # height (percent)
        name: str = rename.get(n := label["rectanglelabels"][0], n)  # rename if the label is exist in rename.
        if name not in labels:
            labels.append(name)
        annotations.append((labels.index(name), x, y, w, h))

    return AnnotationData(annotations, data["image"])


def setup_data(i: int, d: dict[str, Any], labels: list[str], image_map: dict[str, str], rename: dict[str, str]) -> None:
    try:
        role: str = "valid" if i % 5 == 0 else "train"  # 1/5 data is for validation
        a = gen_yolo_data(d, labels, rename)
        image_map[a.s3_path] = f"datasets/{role}/images/{i:03}.png"

        with open(f"learn/datasets/{role}/labels/{i:03}.txt", "w", encoding="utf8") as label_file:
            label_file.write("\n".join(" ".join(map(str, v)) for v in a.annotations))
    except Exception as e:
        print(f"error: {i}: {e}")


def main():
    # make directories
    os.makedirs("learn/datasets/train/images", exist_ok=True)
    os.makedirs("learn/datasets/train/labels", exist_ok=True)
    os.makedirs("learn/datasets/valid/images", exist_ok=True)
    os.makedirs("learn/datasets/valid/labels", exist_ok=True)

    # make annotations
    labels: list[str] = []
    image_map: dict[str, str] = {}

    assert len(argv) in (2, 3), "Usage: python preprocess.py <json file path> [json file path of rename map]"
    assert os.path.exists(argv[1]), f"{argv[1]} does not exist"
    with open(argv[1], encoding="utf8") as f:
        data: list[dict[str, Any]] = json.load(f)

    if len(argv) == 3:
        with open(argv[2], encoding="utf-8") as f:
            rename: dict[str, str] = json.load(f)
    else:
        rename: dict[str, str] = {}

    for i, d in enumerate(data):
        setup_data(i, d, labels, image_map, rename)

    # make download info
    with open("learn/download_path.json", mode="w", encoding="utf8") as f:
        json.dump(image_map, f, ensure_ascii=False)

    # make data.yml template
    with open("./learn/data_template.yml", mode="w", encoding="utf8") as f:
        f.write(DATA_YAML_TEMPLATE.format(len(labels), "\n".join([f"  - {label}" for label in labels])))

    # copy files for zip
    shutil.copy("train.py", "learn/train.py")
    shutil.copy("env.toml", "learn/env.toml")
    shutil.copy("postprocess.py", "learn/postprocess.py")
    shutil.copy("run.sh", "learn/run.sh")

    # create zip with necessary files
    with zipfile.ZipFile("learn.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk("learn"):
            zf.mkdir(root, 0o755)
            for file in files:
                zf.write(os.path.join(root, file))

    print("Done!!")


if __name__ == "__main__":
    main()

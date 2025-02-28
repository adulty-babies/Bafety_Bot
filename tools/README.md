# Utility tools
We are using these tools for training.

## preprocess.py
pre process script for training at GPU server.
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

## other files
Assets for preprocess.py.

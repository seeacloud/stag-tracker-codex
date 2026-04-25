# Train YOLO For STag ROI Detection

This project did not train YOLO. The previous tests used the public `yolov8n.pt`
model only to verify that the pipeline works.

If YOLO is needed, train a one-class model that detects only the outer STag marker
or the board that carries the marker.

## Class

Use one class:

```text
stag
```

Do not keep COCO classes. The trained model should output only `stag`.

## Dataset Layout

```text
datasets/stag/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

`data.yaml`:

```yaml
path: datasets/stag
train: images/train
val: images/val
names:
  0: stag
```

Each label file uses YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to 0..1. Example:

```text
0 0.512 0.438 0.184 0.201
```

## What To Label

For this pipeline, label the full visible marker border or the full marker board.
Do not label unrelated objects. If the camera sees the marker at different scales,
angles, exposure levels, motion blur, or partial occlusion, include those cases in
the dataset.

Useful starting point:

```text
300-800 images for a controlled scene
1000-3000 images for different rooms, distances, lighting, and motion
```

## Train

Install YOLO only when you need training or YOLO inference:

```powershell
pip install -r requirements-yolo.txt
```

Train a small model:

```powershell
yolo detect train model=yolov8n.pt data=datasets/stag/data.yaml imgsz=640 epochs=100 batch=16 device=0 name=stag_yolov8n
```

For speed, export TensorRT on the target GPU:

```powershell
yolo export model=runs/detect/stag_yolov8n/weights/best.pt format=engine half=True imgsz=640 device=0
```

Then run:

```powershell
python -m vision_fusion.main --source 0 --yolo-model runs/detect/stag_yolov8n/weights/best.pt --classes 0 --device 0 --detect-interval 10 --show
```

## When Not To Use YOLO

If the only requirement is STag ID, corners, and optional pose, use STag directly:

```powershell
python -m vision_fusion.stag_only --source 0 --show
```

YOLO is useful only when you need a coarse ROI because the scene is cluttered,
the image is large, or marker search over the full frame is too expensive.

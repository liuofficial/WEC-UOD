from ultralytics import YOLO
import torch
torch.use_deterministic_algorithms(False)
# 自己的路径
model_yaml = "/WEC-UOD/ultralytics/cfg/models/11/yolo11-yuan.yaml"
model = YOLO(model_yaml)


model.train(
    data="data_uod.yaml",
    epochs=1000,
    imgsz=640,
    batch=32,
    device=1,      # GPU id
    workers=4,
    project="WEC-UOD/runs",
    name="progress_a",
    exist_ok=True,
    amp=True,
    pretrained=False,
)

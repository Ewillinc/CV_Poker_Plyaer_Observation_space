


from pathlib import Path
import yaml
import torch
import torch.serialization as ts


# Пофикшенная ошибка на жалобу Pytorch c weights_only = False,
# Сейчас пофикшена самим YOLO
"""from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
ts.add_safe_globals([DetectionModel, Sequential])"""


from ultralytics import YOLO



if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]  # для python
else:
    ROOT = Path.cwd().resolve().parents[0]      # для jupyter_notebook(не актульно)


DATA_DIR   = ROOT / "dataset/yolo_det"
YAML_FILE  = DATA_DIR / "poker.yaml"
WEIGHTS    = ROOT / "pretrained" / "yolov8m.pt"   
RUNS_DIR   = ROOT / "pretrained"



with YAML_FILE.open("r", encoding="utf-8") as f:
    names = yaml.safe_load(f)["names"]

model = YOLO(str(WEIGHTS))          
results = model.train(
    data=str(YAML_FILE),
    epochs=60,
    imgsz=640,
    batch=8,
    device=0,                       # первая GPU -> 0 
    workers=8,
    project=str(RUNS_DIR),
    name="poker",     
    exist_ok=True,  
    save=True       
)


metrics = model.val(
    data=str(YAML_FILE),
    split="val",
    device=0,
    imgsz=640,
)

model.model.names = {i: f"{i+1}:{n}" for i, n in enumerate(names)}

model.save(str(RUNS_DIR / "yolov8m_poker.pt"))
print(f"Model save {RUNS_DIR}/yolov8_poker.pt")




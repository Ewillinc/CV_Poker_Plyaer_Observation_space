from Observation_space import get_state, HeroObs, OtherPlayerObs, TableInfo
from torchvision import transforms, datasets
from ultralytics import YOLO
from pathlib import Path
import easyocr
import torch
import timm
import sys
import csv
import os


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd().resolve().parents[0] # для Jupyter

# --- Константы

DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH    = 0.3                       

YOLO_WEIGHTS    = ROOT / 'pretrained/yolov8m_poker.pt'
TEST_DIR        = ROOT / 'test'
EFFICI_WEIGHTS  = ROOT / 'pretrained/efficientnetb0_best.pth'
TRAIN_DIR       = ROOT / "dataset/eff_class/train"

# --- Папки для кропов -----------

OUTPUT_DIR_CARD     = 'OUTPUT/cards'        
OUTPUT_DIR_BETS     = 'OUTPUT/bets'
OUTPUT_DIR_PL_CHIPS = 'OUTPUT/chips'
OUTPUT_DIR_POT      = 'OUTPUT/pot'
OUTPUT_DIR_OCR      = 'OUTPUT/ocr'


dirs = [OUTPUT_DIR_CARD,OUTPUT_DIR_BETS,
        OUTPUT_DIR_PL_CHIPS,OUTPUT_DIR_POT,
        OUTPUT_DIR_OCR]

for d in dirs:   
    os.makedirs(d, exist_ok= True)  # создаем папки если их нет


# ------- CSV ------

META_PATH = os.path.join('OUTPUT', 'metadata.csv')

with open(META_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'image', 'yolo_class', 'x1', 'y1', 'x2', 'y2',
        'effnet_label', 'ocr_text'
    ])

model = YOLO(YOLO_WEIGHTS)

datasets = datasets.ImageFolder(TRAIN_DIR, transform = transforms.ToTensor())
class_names = datasets.classes

NUM_CLASSES = 52
CARD_LABEL = 'card'

cls_model = timm.create_model('efficientnet_b0', pretrained= False, num_classes = NUM_CLASSES )

state_dict = torch.load(EFFICI_WEIGHTS, map_location = DEVICE)
cls_model.load_state_dict(state_dict)

cls_model = cls_model.to(DEVICE).eval()

cls_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])

reader = easyocr.Reader(['ru'],  gpu=torch.cuda.is_available())

OCR_LABELS = {'pot_area',
              'player_chips_area',
              'bets'}



def main():
    """
    Функция просит загрузить изображение
    и преобразует его в вектора состояний
    """

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Введите путь к файлу скриншота покерного стола: ").strip()
    if not image_path:
        print("Не указан путь к изображению.")
        return
    hero, others, table = get_state(image_path,
                                    yolo_model = model,
                                    device = DEVICE,
                                    cls_model = cls_model,
                                    cls_transform = cls_transform,
                                    reader = reader,
                                    class_names = class_names)

    print("Hero:", hero)
    print("Other players:", others)
    print("Table:", table)

if __name__ == "__main__":
    main()

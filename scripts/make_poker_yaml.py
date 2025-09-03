from pathlib import Path
import yaml



ROOT = Path(__file__).resolve().parents[1] / "dataset/yolo_det"



CLASS = [
    "pot_area",
    "player_area",
    "player_chips_area",
    "check",
    "fold",
    "call",
    "raise",
    "fold_player_area",
    "bets",
    "self_player_area",
    "0",             
    "card",
]


yaml_dict = {
    "train": str((ROOT / "images/train").resolve()),
    "val"  : str((ROOT /"images/val").resolve()),
    "test" : str((ROOT / "images/test").resolve()),
    "nc"   : len(CLASS),
    "names": CLASS,
}


out_yaml = ROOT / "poker.yaml"

out_yaml.parent.mkdir(parents= True, exist_ok= True)
yaml.safe_dump(yaml_dict, out_yaml.open("w"), allow_unicode= True)
print(f'{ROOT} <--- YAML created.')
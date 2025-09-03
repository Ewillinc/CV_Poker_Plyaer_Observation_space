
import numpy as np
from PIL import Image
import torch
import math
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional




def observation_space(yolo_model,
                              device,
                              cls_model,
                              cls_transform,
                              reader,       
                              image_path, 
                              class_names,
                        
                              first_round=False, 
                              initial_player_stacks=None):
    """
    Извлекает вектор состояния (observation_space) из скриншота покерного стола.
    
    Параметры:
      - yolo_model
      - cls_model
      - cls_transform
      - reader 
      - image_path(str): путь к изображению скриншота.
      - class_names: названия всех классов
      - first_round (bool): флаг первого раунда (True, если это начало новой раздачи, ещё не реализован).
      - initial_player_stacks (list или None): начальные стеки игроков в начале раздачи  для сравнения (не реализован).
    """

    yolo_model = yolo_model
    device = device 
    cls_model = cls_model
    class_names = class_names
    cls_transform = cls_transform
    reader = reader

    img = Image.open(image_path).convert('RGB')



    results = yolo_model.predict(source=img, device=device, imgsz=640, conf=0.3)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)   
    class_ids = results.boxes.cls.cpu().numpy().astype(int) 
    class_name_map = results.names 
    dets = {}

    print(dets)

    for i, cid in enumerate(class_ids):
        
        raw_label = class_name_map[cid]     
        label = raw_label.split(':', 1)[1]   
        x1, y1, x2, y2 = boxes[i]
       
        crop = img.crop((x1, y1, x2, y2))
        ocr_text = ""   
        eff_label = ""  

        # ЕСЛИ РАСПОЗНАННЫЙ ОБЪЕКТ КАРТА:
        if label == "card":
           
            inp = cls_transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = cls_model(inp)
                pred_idx = int(output.argmax(dim=1).item())
           
            eff_label = class_names[pred_idx]

        # ЕСЛИ РАСПОЗНАННЫЙ ОБЪЕКТ СОДЕРЖИТЬ СИМВОЛЫ ДЛЯ ДЕТЕКЦИИ OCR

        elif label in {"pot_area", "player_chips_area", "bets"}:

            
            text_list = reader.readtext(np.array(crop), detail=0)

            pattern = r'\d+(?:,\d{1,2})?' 
            text = ' '.join(text_list) 
            m = re.search(pattern, text)
            num = float(m.group().replace(',', '.')) if m else None
            if text_list:
                ocr_text = num # берем первую распознанную строку, если есть
            else:
                ocr_text = ""  # если OCR ничего не вернул


        
        key = f"{label}_{i}"
        dets[key] = [x1, y1, x2, y2, ocr_text, eff_label]

        """
                    Получается словарь следующего вида:
        data = {
        "card_0"        : [x1,y1,x2,y2, "",  "51"],     51 - туз пик,
        "pot_area_1"    : [x1,y1,x2,y2, "1500", ""],    15 - BB где 15 *100, размер BB в RB_model
        "player_area_2" : [x1,y1,x2,y2,"", ""]    ,
        ...
        }
        """

        #------------------OBSERVATION_SPACE_VECTOR------------------------
        """Из полученного вектора будем собирать player observation space vector"""


    return dets

"""
Структурирование состояния покерного стола (устойчивая версия).
  1 Находим центр стола.
  2 Собираем кандидатов сидений по всем «сидельным» сигналам.
  3 Кластеризуем кандидатов по углу (на окружности) → получаем реальные сидения.
  4 Привязываем к кластерам (сидениям) стеки/ставки/статусы.
  5 Героя определяем по self_player_area и действиям; его карты и действия — отдельно.
  6 Формируем Hero, OtherPlayers, TableInfo.

"""


# Структуры для хранения данных

@dataclass  # Упрощенный декоратор для создания клссов для хранения данных с авто __init__
class HeroObs:
    """Наблюдения по self player"""
    seat_id: Optional[int] = None          # порядковый номер self sit в обходе по часовой
    chips: float = 0.0                     # стек 
    bet: float = 0.0                       # текущая ставка (если есть отдельный объект bets_* для героя)
    status: str = "playing"                # playing | fold
    cards: List[int] = field(default_factory=list)  # индексы двух карт героя
    actions: Dict[str, int] = field(default_factory=lambda:  # маска доступных действий
                                     {"fold": 0, "check": 0, "call": 0, "raise": 0})

@dataclass
class OtherPlayerObs:
    """Other Player Observation."""
    seat_id: int = -1
    chips: float = 0.0
    bet: float = 0.0
    status: str = "unknown"                # playing | fold | unknown

@dataclass
class TableInfo:
    """Общая информация"""
    community_cards: List[int] = field(default_factory=list)
    pot: float = 0.0

# Геометрия для сопоставления детекции к хранилищам.

def box_center(box: Tuple[float, float, float, float]):
    """
    Вычисляем центр прямоугольника детекции(center_x, center_y)
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def box_area(box: Tuple[float, float, float, float]):
    """
    Площадь прямоугольника (учитывает вырожденные случаи)
    """
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersection(a: Tuple[float, float, float, float],
                 b: Tuple[float, float, float, float]):
    """
    Площадь пересечения двух прямоугольников
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

def coverage(inner: Tuple[float, float, float, float],
             outer: Tuple[float, float, float, float]):
    """
    Доля площади inner, покрытая outer (0..1)
    Нужна для «мягкой» принадлежности (если объект частично попал в область)
    """
    a = box_area(inner)
    return (intersection(inner, outer) / a) if a > 0 else 0.0

def angle_of_point(center: Tuple[float, float], p: Tuple[float, float]):
    """
    Полярный угол точки p относительно center
    """
    cx, cy = center
    px, py = p
    ang = math.atan2(py - cy, px - cx)         
    return (ang + 2 * math.pi) % (2 * math.pi) 

def circ_dist(a: float, b: float) -> float:
    """
    Кратчайшая круговая дистанция между углами a и b (в радианах)
    Для поиска ближайшего кластера на окружности
    """
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)

#  Оценка центра стола

def estimate_table_center(detections: Dict[str, list]) -> Tuple[float, float]:
    """
    Центр берём из pot_area_* (самый стабильный якорь).
    Если пота нет(что мы считаем x-> не вероятным),
    усредняем центры всех сидельных признаков:
    player_area_*  fold_player_area_*  player_chips_area_* self_player_area_*
    может быть устойчева даже при пропуске части сидений.
    """
    # Поиск Pot_area
    pot_boxes = [v[:4] for k, v in detections.items() if k.startswith("pot_area")]
    if pot_boxes:
        return box_center(pot_boxes[0])  

    # Альтернативные маркеры
    anchors = []
    for k, v in detections.items():
        if (k.startswith("player_area") or k.startswith("fold_player_area") or
            k.startswith("player_chips_area") or k.startswith("self_player_area")):
            anchors.append(v[:4])

    if not anchors:
        raise ValueError("Не было обнаружено Pot и player для определения центра стола," \
        "def estimate_table_center Error")

    # Усредняем центры всех найденных якорей
    cxs = []
    cys = []
    for b in anchors:
        cx, cy = box_center(b)
        cxs.append(cx)
        cys.append(cy)
    return sum(cxs) / len(cxs), sum(cys) / len(cys)


#  Кандидаты сидений 

def collect_seat_candidates(detections: Dict[str, list],
                            center: Tuple[float, float]):
    """
    Формируем список кандидатов сидений как пары (angle, weight).
    Берём ВСЕ сигналы, которые «сидят» на окружности:
      player_area_* (вес 1.0), fold_player_area_* (0.9),
      player_chips_area_* (0.6), self_player_area_* (1.2)
    Чем выше вес, тем сильнее вклад в кластерном усреднении.
    """
    cands: List[Tuple[float, float]] = []

    for k, v in detections.items():
        box = v[:4]
        ang = angle_of_point(center, box_center(box))

        if k.startswith("self_player_area"):
            cands.append((ang, 1.2))
        elif k.startswith("player_area"):
            cands.append((ang, 1.0))
        elif k.startswith("fold_player_area"):
            cands.append((ang, 0.9))
        elif k.startswith("player_chips_area"):
            cands.append((ang, 0.6))

    if not cands:
        raise ValueError("Нет кандидатов сидений" \
        "def collect_seat_candidates Error")

    return cands

# кластеризация по углу

def circular_cluster_angles(cands: List[Tuple[float, float]],
                            eps_deg: float = 20.0):
    """
    Простая 1D-кластеризация на окружности по углу:
      1 сортируем кандидатов по углу;
      2 объединяем соседей, если разрыв по углу <= eps_deg;
      3 центр кластера считаем по взвешенному среднему на окружности
         (через сумму векторов sin/cos).
    Возвращаем список углов центров кластеров (в радианах), отсортированных по возрастанию.
    eps_deg подобрали 20, посчитав его устойчивым даже к максимальному(10) количеству игроков.
    """
    if not cands:
        return []

    # Сортируем по углу
    cands = sorted(cands, key=lambda t: t[0]) 

    # Используем радианы
    eps = math.radians(eps_deg)

    clusters: List[List[Tuple[float, float]]] = [[]]  
    clusters[0].append(cands[0])

    # Проходим по cands и склеиваем по порогу eps!
    for i in range(1, len(cands)):
        prev_ang = cands[i - 1][0]
        ang = cands[i][0]
        if (ang - prev_ang) <= eps:
            clusters[-1].append(cands[i])     # тот же кластер
        else:
            clusters.append([cands[i]])       # новый кластер

    # Проверяем «круговое» слияние: первый и последний кластера!
    first_ang = clusters[0][0][0]
    last_ang = clusters[-1][-1][0]
    if ((first_ang + 2 * math.pi) - last_ang) <= eps:
       
        clusters[0] = [(a if a >= first_ang else a + 2 * math.pi, w)
                       for (a, w) in clusters[0]]
        clusters[0].extend([(a if a >= last_ang else a + 2 * math.pi, w)
                            for (a, w) in clusters[-1]])
        clusters.pop()  


    """я умер пока искал помощи с этими функциями"""
    def cluster_center(cluster: List[Tuple[float, float]]):
        sum_sin = 0.0
        sum_cos = 0.0
        for a, w in cluster:
            sum_sin += math.sin(a) * w
            sum_cos += math.cos(a) * w
        ang = math.atan2(sum_sin, sum_cos) 
        return (ang + 2 * math.pi) % (2 * math.pi)

    centers = [cluster_center(c) for c in clusters]
    centers.sort()  # по часовой стрелке
    return centers



# Сбор векторов из Observation_space
def build_state(detections: Dict[str, list],
                eps_deg: float = 20.0
               ):
    """
    Главный конвейер:
      — центр стола → кандидаты сидений → угловые кластеры (реальные сидения)
      — определяем к какому кластеру относится каждый объект (по ближайшему углу)
      — формируем Hero/Other/Table
    """
    """
    Должна будет превратиться в вектор состояний для передачи в будущие модели,
    На данный момень собирает информацию в разные контейнеры.
    """

    # Центр стола 
    center = estimate_table_center(detections)

    # Кандидаты сидений и их угловая кластеризация
    cands = collect_seat_candidates(detections, center)
    seat_angles = circular_cluster_angles(cands, eps_deg=eps_deg)  # список углов сидений
    seat_count = len(seat_angles)                                  # число сидений определяем данными


    #    Храним промежуточные агрегаты по каждому сидению (индекс = порядковый номер в seat_angles)
    seats: List[Dict] = [
        {"chips": 0.0, "bet": 0.0, "has_fold": False, "has_pa": False}
        for _ in range(seat_count)
    ]

    #  Найдём self и его «угол» (обычно статично по центру снизу, но есть разные игровые столы)
    self_box: Optional[Tuple[float, float, float, float]] = None
    for k, v in detections.items():
        if k.startswith("self_player_area"):
            self_box = v[:4]
            break
        
    hero_angle: Optional[float] = angle_of_point(center, box_center(self_box)) if self_box else None

    # Функция для привязки любого бокса к ближайшему сидению по углу

    def nearest_seat_id(box: Tuple[float, float, float, float]) -> int:
        cx, cy = box_center(box)                         # центр бокса
        ang = angle_of_point(center, (cx, cy))           # его угол
        # находим сидение с минимальной круговой дистанцией по углу
        best_i, best_d = 0, float("inf")
        for i, a in enumerate(seat_angles):
            d = circ_dist(ang, a)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    # Герой (карты,действия,стек)
    hero = HeroObs()
    # Объекты стола (pot, общие карты) 
    table = TableInfo()

    # Первый проход: отметить присутствие player_area/fold_area по сидениям
    for label, vals in detections.items():
        box = tuple(vals[:4])
        sid = nearest_seat_id(box)  

        if label.startswith("player_area"):
            seats[sid]["has_pa"] = True
        elif label.startswith("fold_player_area"):
            seats[sid]["has_fold"] = True

    # Второй проход: собрать количественные значения (pot/chips/bets), карты и действия
    for label, (x1, y1, x2, y2, val_chips, val_card) in detections.items():
        box = (x1, y1, x2, y2)
        sid = nearest_seat_id(box)  # привязка к сидению по углу

        # Пот
        if label.startswith("pot_area"):
            if str(val_chips).strip() != "":
                table.pot = float(val_chips)
            continue

        #  Карты
        if label.startswith("card_"):
            # Карты игрока должны находиться в области этого игрока
            if self_box and coverage(box, self_box) >= 0.6: # порог 60% карты в области
                hero.cards.append(int(val_card))
                hero.seat_id = sid                         # фиксируем self
            else:
                table.community_cards.append(int(val_card))
            continue

        # Стек игроков
        if label.startswith("player_chips_area"):
            # Стек игрока, если покрытие self больше порога 40% 
            if self_box and coverage(box, self_box) >= 0.4: 
                hero.chips = max(hero.chips, float(val_chips or 0.0))
                hero.seat_id = sid                         # фиксируем сид игрока
            else:
                seats[sid]["chips"] = max(seats[sid]["chips"], float(val_chips or 0.0))
            continue

        # Ставки (если на изображении рядом с игроком есть фишки)
        if label.startswith("bets_"):
            seats[sid]["bet"] = max(seats[sid]["bet"], float(val_chips or 0.0))
            # если ставка принадлежит сидению  — удержим её и в hero
            if hero.seat_id is not None and sid == hero.seat_id:
                hero.bet = max(hero.bet, float(val_chips or 0.0))
            continue

        #  Доступные действия текущего игрока (детекция кнопок с действиями)
        if label.startswith("fold_"):
            hero.actions["fold"] = 1
            hero.seat_id = hero.seat_id if hero.seat_id is not None else sid
            continue
        if label.startswith("check_"):
            hero.actions["check"] = 1
            hero.seat_id = hero.seat_id if hero.seat_id is not None else sid
            continue
        if label.startswith("call_"):
            hero.actions["call"] = 1
            hero.seat_id = hero.seat_id if hero.seat_id is not None else sid
            continue
        if label.startswith("raise_"):
            hero.actions["raise"] = 1
            hero.seat_id = hero.seat_id if hero.seat_id is not None else sid
            continue


    if hero.seat_id is None and hero_angle is not None:

        best_i, best_d = 0, float("inf")
        for i, a in enumerate(seat_angles):
            d = circ_dist(hero_angle, a)
            if d < best_d:
                best_i, best_d = i, d
        hero.seat_id = best_i

    # Остальные игроки и их статусы:
    others: List[OtherPlayerObs] = []
    for sid in range(seat_count):
        # скипаем сидение игрока
        if hero.seat_id is not None and sid == hero.seat_id:
            continue

        d = seats[sid]
        if d["has_pa"] or d["bet"] > 0:
            status = "playing"
        elif d["has_fold"]:
            status = "fold"
        else:
            status = "unknown"

        others.append(OtherPlayerObs(seat_id=sid, chips=d["chips"], bet=d["bet"], status=status))

    #  Сортируем по порядку обхода 
    others.sort(key=lambda o: o.seat_id)

    return hero, others, table





def get_state(image_path: str,yolo_model,
                              device,
                              cls_model,
                              cls_transform,
                              reader,
                              class_names):
    """
    Выполняет полную обработку скриншота: детекция + сбор состояния
    Возвращает кортеж (HeroObs, список OtherPlayerObs, TableInfo) для переданного изображения
    """
    # image_path (str): путь к файлу скриншота покерного стола
    detections = observation_space(yolo_model, device, cls_model,cls_transform, reader, image_path, class_names)
    return build_state(detections)
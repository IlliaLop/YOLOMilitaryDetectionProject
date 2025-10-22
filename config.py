import numpy as np

class CFG:
    WEIGHTS = 'runs/detect/train/weights/best.pt'
    CONFIDENCE = 0.40
    CONFIDENCE_INT = int(round(CONFIDENCE * 100, 0))
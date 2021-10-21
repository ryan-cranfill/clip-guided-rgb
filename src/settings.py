import torch
from pathlib import Path

THIS_FILE = Path(__file__).absolute()
SRC_DIR = THIS_FILE.parent
ROOT_DIR = SRC_DIR.parent
MODEL_DIR = ROOT_DIR / 'models'

CLIP_MODEL_PATH = MODEL_DIR / 'ViT-B-32.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

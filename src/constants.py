from pathlib import Path
from enum import Enum

class Constants(Enum):
    VERSION = "C_0"
    USE_CASE = "Land_Cover_Semantic_Segmentation"
    CONFIG_PATH = Path("config/config.yaml")
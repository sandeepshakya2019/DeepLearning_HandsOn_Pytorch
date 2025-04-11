# All imports (verify they work)
print("\n🔽 Importing packages...\n")

import os
import sys
import cv2
import torch
import random
import requests
import shutil
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

print("\n✅ All packages installed and imported successfully!")
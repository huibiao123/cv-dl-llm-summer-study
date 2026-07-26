import cv2
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent   
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "output"             

image_paths = [
 IMAGE_DIR / "anime.jpg",
 IMAGE_DIR / "lenna.jpg",
 IMAGE_DIR / "scenery.jpg"
]

for path in image_paths:
 img = cv2.imread(str(path))
 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 resize = cv2.resize(img,None,fx=0.5,fy=0.5)

 filename = os.path.basename(path)
 name, ext = os.path.splitext(filename)

 cv2.imwrite(str(OUTPUT_DIR / f"{name}_gray{ext}"),gray)
 cv2.imwrite(str(OUTPUT_DIR / f"{name}_hsv{ext}"),hsv)
 cv2.imwrite(str(OUTPUT_DIR / f"{name}_resize{ext}"),resize)

 print(path)
 print(img.shape)
 print(img.min(),img.max())
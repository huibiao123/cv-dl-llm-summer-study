import cv2
import os

image_paths = [
    "images/anime.jpg",
    "images/lenna.jpg",
    "images/scenery.jpg"
]


for path in image_paths:

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #111
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    resize = cv2.resize(img,None,fx=0.5,fy=0.5)
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    cv2.imwrite(f"output/{name}_gray{ext}",gray)
    cv2.imwrite(f"output/{name}_hsv{ext}",hsv)
    cv2.imwrite(f"output/{name}_resize{ext}",resize)
    print(path)
    print(img.shape)
    print(img.min())
    print(img.max())
    


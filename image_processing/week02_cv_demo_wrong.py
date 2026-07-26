import cv2
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "output" 

image_path = IMAGE_DIR / "coins.png"


img = cv2.imread(str(image_path))

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


blur = cv2.GaussianBlur(
    gray,
    (5,5),
    0
)


_, thresh = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY
)



kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)


closed = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel
)



contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)



result = img.copy()


count = 0


for contour in contours:


    area = cv2.contourArea(contour)


    if area > 500:


        count += 1


        cv2.drawContours(
            result,
            [contour],
            -1,
            (0,255,0),
            2
        )


        M = cv2.moments(contour)


        cx = int(
            M["m10"]/M["m00"]
        )

        cy = int(
            M["m01"]/M["m00"]
        )


        cv2.putText(
            result,
            str(count),
            (cx,cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )



print(
    "检测数量:",
    count
)


result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(10,8))
plt.imshow(result_rgb)
plt.axis("off")
plt.savefig(str(OUTPUT_DIR / "result_wrong.png"), bbox_inches="tight", pad_inches=0)
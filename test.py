"""test.py: Main code to train parking slots monitor"""

# Example Usage: ./cv_hw1 -i image -k clusters -m grey
# Example Usage: python cv_hw1 -i image -k clusters -m rgb


__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

import cv2
​
face = cv2.CascadeClassifier("./data/cascade.xml")  # 人脸识别
cap = cv2.VideoCapture(2)
​
while True:
    ret, img = cap.read()
    roi_color = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图片，haar特征都是在灰度上检测
    faces = face.detectMultiScale(gray, 1.3, 5)  # 参数1：灰度图片数据   2:缩放比例 3：人脸大小不能小于5个像素
​
# 获取宽高信息，个人脸画方框
for (x, y, w, h) in faces:
    # img：要画的图片，(x,y)：起始坐标 ，(x+w,y+h)：宽高，颜色，线条宽度
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('10', img)
if cv2.waitKey(5) & 0xff == ord("q"):
    break
​
cap.release()
cv2.destroyAllWindows()
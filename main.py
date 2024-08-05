import cv2

import face_recognition as fr

from Images import cam

from datetime import datetime 



current_time = datetime.now().strftime("%Y-%m-%d-%S")

dar = cv2.imread('Images\\2024-08-05-33.jpg')
rgb_img_dar = cv2.cvtColor(dar,cv2.COLOR_BGR2RGB)
dar_en=fr.face_encodings(rgb_img_dar)[0]


su = cv2.imread(cam.cap())
rgb_img_su = cv2.cvtColor(su,cv2.COLOR_BGR2RGB)
su_en=fr.face_encodings(rgb_img_su)[0]

result = fr.compare_faces([dar_en],su_en)
print(result)

cv2.imshow('Img',dar)
cv2.imshow('Img 2',su)

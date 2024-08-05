#import statements 
import os


import cv2

import csv

import numpy as np

from datetime import datetime 

import face_recognition as fr




vid = cv2.VideoCapture(0)

drashan_img = fr.load_image_file('Images\darshani.jpg')
drashan_img_en = fr.face_encodings(drashan_img)[0]

sumu_img = fr.load_image_file('Images\sumu.jpg')
sumu_img_en = fr.face_encodings(sumu_img)[0]


en_list = [drashan_img_en,sumu_img_en]

sname = ['darshani','sumu']

stu = sname.copy()

floc = []

fen = []

fnames = []

s = True


current_time = datetime.now().strftime("%Y-%m-%d")

file = open(current_time+'.csv','w+',newline='')
write = csv.writer(file)
i = 0
while 0<=i:
    i+1    
    _,frame = vid.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_frame = small_frame[:,:,::-1]
    
    if s:
        floc = fr.face_locations(rgb_frame)
        fen = fr.face_landmarks(rgb_frame,floc)
        fnames =[]

        if fen in fen:

            matches = fr.compare_faces(en_list,fen)
            name = ''
            face_dis = fr.face_distance(en_list,fen)
            best_face = np.argmin(face_dis)

            if matches[best_face]:
                name = sname[best_face]

            fnames.append(name)

            if name in sname:
                if name in stu:
                    stu.remove(name)
                    print(stu)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    write.writerow([name,current_time])
    cv2.imshow('attendencs system',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
file.close()



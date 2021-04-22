import cv2
import numpy as np
import dlib
from playsound import playsound
from imutils import face_utils
from scipy.spatial import distance as dist

def calculate_lip(lips):
     dist1 = dist.euclidean(lips[2], lips[6]) 
     dist2 = dist.euclidean(lips[0], lips[4]) 

     LAR = float(dist1/dist2)

     return LAR

counter = 0
lip_LAR = 0.4
lip_per_frame = 30

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for (i, face) in enumerate(faces):
        lips = [60,61,62,63,64,65,66,67]
        point = predictor(gray, face)
        points = face_utils.shape_to_np(point)
        lip_point = points[lips]
        LAR = calculate_lip(lip_point) 

        lip_hull = cv2.convexHull(lip_point)
        cv2.drawContours(frame, [lip_hull], -1, (0, 255, 0), 1)

        if LAR > lip_LAR:
            counter += 1
            print(counter)
            if counter > lip_per_frame:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound("bangun.mp3")
        else:
            counter = 0

    cv2.putText(frame, "LAR: {:.2f}".format(LAR), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    cv2.imshow("yawn detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

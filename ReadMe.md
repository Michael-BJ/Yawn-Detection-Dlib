# How it's works
1. First import the libaray that we need
````
import cv2
import numpy as np
import dlib
from playsound import playsound
from imutils import face_utils
from scipy.spatial import distance as dist
````
2. Connect the webcam to the program 
````
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Blink detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
````
3. Make the program to target or detect the face
````
detector = dlib.get_frontal_face_detector() 
````
4. Insert the model that we need
````
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
````
5. Change the BGR to the grayscale
````
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
````
6. Detect the face in the grayscale image
````
faces = detector(gray)
````
7. Convert the coordinates of facial landmark to a numpy
````
point = predictor(gray, face)
points = face_utils.shape_to_np(point)
````
8. Calculate the coordinates/indexes for the up and down lip
````
def calculate_lip(lips):
     dist1 = dist.euclidean(lips[2], lips[6]) 
     dist2 = dist.euclidean(lips[0], lips[4]) 
````
9. Calculate the ratio of the lip using the coordinates/indexes that have been obtained
````
EAR = (A + B) / (2.0 * C)`
````
10. Determine the lip aspect ratio for yawn and the number of frames the person has yawned
````
counter = 0
lip_LAR = 0.4
lip_per_frame = 30
````
11. Insert the indexes of the lip
````
lips = [60,61,62,63,64,65,66,67]
````
12. Change the indexes into the numpy array 
````
lip_point = points[lips]
````
13. Calculate the lip aspect ratio
````
LAR = calculate_lip(lip_point) 
````

# Demo

Clik the picture to see the Video
[![Watch the video](https://img.youtube.com/vi/QkwzEbPBNz4/maxresdefault.jpg)](https://youtu.be/QkwzEbPBNz4)

from cv2 import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(r'C:\Users\Gaurav\Desktop\Projects\Face Detection\haarcascade_frontalface_default.xml')

# image to detect faces
img = cv2.imread(r'C:\Users\Gaurav\Desktop\Projects\Face Detection\gaurav.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# print(face_coordinates)

# Draw rectangle around the image
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256),randrange(256)), 5)

# Display the images with faces
cv2.imshow('Face Detector', img)
cv2.waitKey()



print("Code Completed")
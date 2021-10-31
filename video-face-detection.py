from cv2 import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(r'C:\Users\Gaurav\Desktop\Projects\Face Detection\haarcascade_frontalface_default.xml')

# Video to detect faces
# webcam = cv2.VideoCapture('video-path')

# For using webcam
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # Read the current frame
    successfull_frame_read, frame = webcam.read()

    # Converting in GrayScale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256),randrange(256)), 5)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if G is pressed
    if key==71 or key==103:
        break

# Release the video CaptureObject
webcam.release()   

print("Code Completed")


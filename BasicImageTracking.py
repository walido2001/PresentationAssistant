# Learning to track eyes using the following link
# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

import cv2
import numpy

# Utilize face and eye classifiers
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

# Testing: importing an image
img = cv2.imread("miscellaneous/randomPhoto.jpg")

# Convert to GrayScale
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Object Detection. Returns objects as list of rectangles
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

# faces contains list of coordinates pertaining to a rectangle. First object would be the face.
for (x,y,w,h) in faces:
    # Draw a rectangle. (image to be drawn on, starting point, ending point, color, border thickness )
    cv2.rectangle( img,(x,y), (x+w,y+h),(255,255,0), 2 )
    
    #Take the face only from the grey-image
    gray_face = gray_picture[y:y+h, x:x+w]

    #Same as above, except in regular color
    face = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(gray_face)

    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0,225,255), 2)

# Open a window
cv2.imshow('my image', img)

# Keep the window open for x amount of secs, if 0 keep it open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()

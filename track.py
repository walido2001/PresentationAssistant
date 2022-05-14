# Learning to track eyes using the following link
# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

import cv2
import numpy

# Utilize face and eye classifiers
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

# Initializing a blob detector
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(detector_params)

# Function that detects and return a face in the image
#   Given that classifiers can return inaccurate objects that it thinks are faces. 
#   The function below returns the object with the largest area. This function/process
#   assumes that the face will take up most of the image, meaning that the face will always be 
#   close to the camera.
def detect_face(img, classifier):
    """Detects a face in the image
    
    Keyword arguments:
    img -- Image containing a face
    classifier -- A face classifier
    Return: An image
    """
    
    greyedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceImage = classifier.detectMultiScale(greyedImage, 1.3, 5)

    if len(faceImage) > 1:
        biggestObject = (0, 0, 0, 0)
        for object in faceImage:
            area = object[2] * object[3]
            bigArea = biggestObject[2] * biggestObject[3]
            if area > bigArea:
                biggestObject = object
    elif len(faceImage) == 1:
        biggestObject = faceImage
    else:
        return None
    
    for (x, y, w, h) in biggestObject:
        face = img[y:y+h, x:x+w]
    return face

# Function that returns 2 values containing images of each eye
#   Given that eye classifiers are sometimes inaccurate. The algorithm below returns the objects that 
#   are above and on the sides of the center of the image. Given that eyes ussually fall above the 
#   half of the face (if cut horizentally) and on either side (if cut vertically).
def detect_eyes(img, classifier):
    """Detects eyes of a face image
    
    Keyword arguments:
    img -- image of a face
    classifier -- an eye classifier
    Return: 2 images of eyes
    """
    
    greyedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detectedEyes = classifier.detectMultiScale(greyedImage, 1.3, 5)

    faceWidth = numpy.size(img, 1)
    faceHeight = numpy.size(img, 0)

    # faceHeight = img.shape[0]
    # faceWidth = img.shape[1]

    leftEye = None
    rightEye = None
    for (x, y, w, h) in detectedEyes:
        if y > faceHeight/2:
            pass

        eyeCenter = (x+w)/2
        if eyeCenter < faceWidth/2:
            leftEye = img[y:y+h, x:x+w]
        else:
            rightEye = img[y:y+h, x:x+w]
        
        return leftEye, rightEye

# Function that cuts out an eyebrow from the image that contains an eye
def cut_eyebrow(img):
    """Cuts out the brow from an eye image
    
    Keyword arguments:
    img -- an Image of an eye
    Return: void
    """
    
    height = img.shape[0]
    width = img.shape[1]
    eyebrowHeight = int(height/4)
    img = img[eyebrowHeight:height, :width]

def blobbing(img, threshold, detector):
    greyedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding: the first returned value is the threshold used, the second is the image
    #   Parameters: image | threshold | maxColorVal | thresholdingType (Check docs)
    _, thresholdedImage = cv2.threshold(greyedImage, threshold, 255, cv2.THRESH_BINARY)

    # Extra modifications that make the pupil stand out more
    thresholdedImage = cv2.erode(thresholdedImage, None, iterations=2) 
    thresholdedImage = cv2.dilate(thresholdedImage, None, iterations=4)
    thresholdedImage = cv2.medianBlur(thresholdedImage, 5)

    keyPoints = detector.detect(thresholdedImage)
    return keyPoints

def doNothing(val):
    return None

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, doNothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            if eyes is not None:
                for eye in eyes:
                    if eye is not None:
                        threshold = cv2.getTrackbarPos('threshold', 'image')
                        eye = cut_eyebrow(eye)
                        keypoints = blobbing(eye, threshold, detector)
                        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()
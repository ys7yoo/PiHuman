# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    print(image.shape)
    # image = imutils.resize(image, width=min(400, image.shape[1]))

    # convert to gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # plot detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)


    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} faces".format(filename, len(faces)))

    # show the output images
    cv2.imshow("face detection", image)
    cv2.waitKey(0)

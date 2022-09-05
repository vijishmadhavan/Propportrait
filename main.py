from google.colab.patches import cv2_imshow
from PIL import Image
import imutils
import cv2
import numpy as np
import imquality.brisque as brisque
import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
sys.path.append("/Proppotrait/")
from angle import face_angle 
from face import eye_blink


ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())



caffemodel = "./checkpoint/Widerface-RetinaFace.caffemodel"
deploy = "./checkpoint/deploy.prototxt"
#eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_eye.xml")
opencv_dnn_model  = cv2.dnn.readNetFromCaffe(deploy, caffemodel)


def cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.75, display = True):

    image = imutils.resize(image, width=600)
    h, w, _ = image.shape



    output_image = image.copy()
    
    preprocessed_image = cv2.dnn.blobFromImage(image, 1, mean=(104, 117, 123))

    opencv_dnn_model.setInput(preprocessed_image)
    
    results = opencv_dnn_model.forward()    
    a=[]
    for i in range(0, results.shape[2]):
      confidence = results[0, 0, i, 2]
      if confidence > min_confidence:
          box = results[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
          a.append(box)
    if len(a) > 1:
      print("More than one person")
    else:
      frame = output_image[startY:endY, startX:endX]
      #cv2_imshow(frame)
      eye_blink(frame)
      b = face_angle(frame)
      if b != None:
        print(b)
      elif b not in ['Looking Left','Looking Right','Looking Down','Looking Straight']:
        print('head not facing camera')
      imgg = Image.fromarray(frame)
      #print(brisque.score(imgg))
      if brisque.score(imgg)<26 :
        print("Not Blurry")
      else:
        print("blurry")

image = cv2.imread(args["image"])
cvDnnDetectFaces(image, opencv_dnn_model, display=True)

import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from PIL import Image
import argparse
import imquality.brisque as brisque
import sys
sys.path.append("/Proppotrait/")
from network import Network


ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())



def load_snapshot(model, snapshot):
    saved_state_dict = torch.load(snapshot,map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)

def headpose(frame, min_confidence=0.75, display = True):
  #face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
  caffemodel = "./checkpoint/Widerface-RetinaFace.caffemodel"
  deploy = "./checkpoint/deploy.prototxt"
  opencv_dnn_model  = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')  # eye cascade
  pose_estimator = Network(bin_train=False)
  load_snapshot(pose_estimator,"./checkpoint/model-b66.pkl")
  pose_estimator = pose_estimator.eval()
  h, w, _ = frame.shape
  preprocessed_image = cv2.dnn.blobFromImage(frame, 1, mean=(104, 117, 123))
  output_image = frame.copy()
  opencv_dnn_model.setInput(preprocessed_image)
  results = opencv_dnn_model.forward()    
  a=[]
  face_tensors=[]
  for i in range(0, results.shape[2]):
    confidence = results[0, 0, i, 2]
    if confidence > min_confidence:
        box = results[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        a.append(box)
  if len(a) > 1:
    print("More than one person")
  else:
    image = output_image[startY:endY, startX:endX]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  transform_test = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
  eyes = eye_cascade.detectMultiScale(gray) 
  if len(eyes)>0:
    print("eye open")
  else:
    print("eye closed")
  if brisque.score(image)<26 :
    print("Not Blurry")
  else:
    print("blurry")
  pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(image,(224,224)), cv2.COLOR_BGR2RGB))
  face_tensors.append(transform_test(pil_img)[None])
  face_tensors = torch.cat(face_tensors,dim=0)
  roll, yaw, pitch = pose_estimator(face_tensors)
  for img, r,y,p in zip(image, roll,yaw,pitch):
    headpose = [r,y,p]
    if headpose[2].item() > 10:
      print("looking up")
    elif headpose[2].item() < -10:
      print("looking down")
    elif headpose[1].item() > 10:
      print("looking right")
    elif headpose[1].item() < -10:
      print("looking left")
    elif headpose[0].item() > 10:
      print("Tilted right")
    elif headpose[0].item() < -10:
      print("Tilted left")
    else:
      print("looking straight")


frame = cv2.imread(args["image"])

headpose(frame)

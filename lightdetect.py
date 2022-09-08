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

def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x= max(bbox[0] + bbox[2]/2 - w/2,0)
    y= max(bbox[1] + bbox[3]/2 - w/2,0)
    return np.asarray([x,y,w,w],np.int64)

def headpose(frame):
  face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
  pose_estimator = Network(bin_train=False)
  load_snapshot(pose_estimator,"./models/model-b66.pkl")
  pose_estimator = pose_estimator.eval()
  #frame = cv2.imread(path)
  gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


  transform_test = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
  faces = face_cascade.detectMultiScale(gray_img, 1.2)
  if len(faces) > 1:
    print("more than one person")
  else:
    face_tensors = []
    face_images = []
    for i, bbox in enumerate(faces):
      x,y, w,h = scale_bbox(bbox,1.5)
      frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2)
      face_img = frame[y:y+h,x:x+w]
      if brisque.score(face_img)<26 :
        print("Not Blurry")
      else:
        print("blurry")
      face_images.append(face_img)
      pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB))
      face_tensors.append(transform_test(pil_img)[None])
      face_tensors = torch.cat(face_tensors,dim=0)
      roll, yaw, pitch = pose_estimator(face_tensors)
      for img, r,y,p in zip(face_images, roll,yaw,pitch):
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

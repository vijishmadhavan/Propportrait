from google.colab.patches import cv2_imshow
from PIL import Image
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
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

def detect(image):
  mp_face_detection = mp.solutions.face_detection
  a =[]
  # For static images:
  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5) as face_detection:
      image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_rows, image_cols, _ = image.shape
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(image_input)
      #annotated_image = image.copy()

      for face_no, face in enumerate(results.detections):
        a.append(face_no)

  if len(a) > 1:
    print("more than one person")
  else:
    detection=results.detections[0]

    location = detection.location_data

    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
    image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
    image_rows)
    xleft,ytop=rect_start_point
    xright,ybot=rect_end_point
    frame = image[ytop: ybot, xleft: xright]
    eye_blink(frame)
    b = face_angle(frame)
    if b != None:
      print(b)
    elif b not in ['Looking Left','Looking Right','Looking Down','Looking Straight']:
      print('head not facing camera')
    imgg = Image.fromarray(frame)
    if brisque.score(imgg)<26 :
      print("Not Blurry")
    else:
      print("blurry")

image = cv2.imread(args["image"])
cvDnnDetectFaces(image, opencv_dnn_model, display=True)

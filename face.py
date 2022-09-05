
import mediapipe as mp
import cv2
import math
def eye_blink(frame):
  map_face_mesh = mp.solutions.face_mesh
  RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
  LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
  CLOSED_EYES_FRAME =3


  def landmarksDetection(img, results, draw=False):
      img_height, img_width= img.shape[:2]
      # list[(x,y), (x,y)....]
      mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

      if draw :
          [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

      # returning the list of tuples for each landmark 
      return mesh_coord

  def euclaideanDistance(point, point1):
      x, y = point
      x1, y1 = point1
      distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
      return distance

  # Right eyes
  
      # horizontal line 
      rh_right = landmarks[right_indices[0]]
      rh_left = landmarks[right_indices[8]]
  # vertical line 
      rv_top = landmarks[right_indices[12]]
      rv_bottom = landmarks[right_indices[4]]

  # Blinking Ratio
  def blinkRatio(img, landmarks, right_indices, left_indices):
      # Right eyes 
      # horizontal line 
      rh_right = landmarks[right_indices[0]]
      rh_left = landmarks[right_indices[8]]
      # vertical line 
      rv_top = landmarks[right_indices[12]]
      rv_bottom = landmarks[right_indices[4]]
      # draw lines on right eyes 
      # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
      # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)
      # LEFT_EYE 
      # horizontal line 
      lh_right = landmarks[left_indices[0]]
      lh_left = landmarks[left_indices[8]]
      # vertical line 
      lv_top = landmarks[left_indices[12]]
      lv_bottom = landmarks[left_indices[4]]
      # Finding Distance Right Eye
      rhDistance = euclaideanDistance(rh_right, rh_left)
      rvDistance = euclaideanDistance(rv_top, rv_bottom)
      # Finding Distance Left Eye
      lvDistance = euclaideanDistance(lv_top, lv_bottom)
      lhDistance = euclaideanDistance(lh_right, lh_left)
      # Finding ratio of LEFT and Right Eyes
      reRatio = rhDistance/rvDistance
      leRatio = lhDistance/lvDistance
      ratio = (reRatio+leRatio)/2
      return ratio

  with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    #image = cv2.imread('/content/38.jpg')

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
        #print(ratio)
        if ratio >5.2:
          print("eye-closed")
        else:
          print("eyeopen")

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
from imutils.video import FPS
from imutils.video import VideoStream
from PIL import Image
import imutils
import time
import numpy as np
import math
import warnings

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = ''
    default_model = 'detect_edgetpu.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        orig = frame.copy()

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        
        cv2.circle(orig, (275, 445), 390, (0, 0, 255), 3, 8, 0)
        cv2.circle(orig, (275, 445), 365, (0, 0, 255), 3, 8, 0)
        cv2.circle(orig, (275, 445), 340, (0, 0, 255), 3, 8, 0)	
        cv2.circle(orig, (275, 445), 315, (0, 0, 255), 3, 8, 0)
        cv2.circle(orig, (275, 445), 290, (0, 0, 255), 3, 8, 0)
        cv2.circle(orig, (275, 0), 5, (0, 0, 255), 3, 8, 0)
    
        # loop over the results
        for r in objs:
          # extract the bounding box and box and predicted class label
          box = r.bbox
          startX = int(r.bbox.xmin)
          startY = int(r.bbox.ymin)
          endX = int(r.bbox.xmax)
          endY = int(r.bbox.ymax)
          label = labels[r.label_id]
          
          smallY = 500
          midY = 400
          largeY = 300
          
          smallRadius = 310
          midRadius = 375
          largeRadius = 450
          
          if True:
            centerX = ((endX - startX) // 2) + startX
            centerY = ((endY - startY) // 2) + startY
            
            cv2.circle(orig, (250, 25), 10, (0, 0, 255), -1)
            myDistance = int(math.sqrt(((centerX - 275)**2)+((centerY - 445)**2)))
            print(myDistance);
            
            # draw the bounding box and label on the image
            angle = int(math.atan((centerY - 370)/(centerX - 250))*180/math.pi)
            
            if angle == 90: 
              angle = 0
              
            if angle < 0:
              angle = angle + 180
              
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0))
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(orig, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)      
            
            cv2.circle(orig, (centerX, centerY), 5, (0, 0, 255), -1)
            cv2.circle(orig, (250, 370), 5, (0, 0, 255), -1)
            cv2.line(orig, (centerX, centerY), (250, 370), (0, 0, 255), 1)
            cv2.putText(orig, str(angle), (260, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
  
# ----------------------------------------------
# Yolo Keras Face Detection from WebCamera
# ----------------------------------------------

from datetime import datetime
import argparse
import numpy as np
import sys, getopt
import mediapipe as mp 
import cv2
import os
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing import image
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Replace deprecated eager execution check
is_eager = tf.compat.v1.executing_eagerly_outside_functions()

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
model_path = "pretrain\selfie_multiclass.tflite"

#Argument Parser
parser=argparse.ArgumentParser()
parser.add_argument('-I','--image')
parser.add_argument('-V','--video')
args=parser.parse_args()

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)


#YOLOV2
#reference from https://github.com/experiencor/keras-yolo2
# https://github.com/experiencor/keras-yolo2/blob/master/LICENSE
	
def interpret_output_yolov2(output, img_width, img_height):
	anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

	netout=output
	nb_class=1
	obj_threshold=0.4
	nms_threshold=0.3

	grid_h, grid_w, nb_box = netout.shape[:3]

	size = 4 + nb_class + 1;
	nb_box=5

	netout=netout.reshape(grid_h,grid_w,nb_box,size)

	boxes = []
	
	# decode the output by the network
	netout[..., 4]  = _sigmoid(netout[..., 4])
	netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
	netout[..., 5:] *= netout[..., 5:] > obj_threshold

	for row in range(grid_h):
		for col in range(grid_w):
			for b in range(nb_box):
				# from 4th element onwards are confidence and class classes
				classes = netout[row,col,b,5:]
				
				if np.sum(classes) > 0:
					# first 4 elements are x, y, w, and h
					x, y, w, h = netout[row,col,b,:4]

					x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
					y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
					w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
					h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
					confidence = netout[row,col,b,4]
					
					box = bounding_box(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
					
					boxes.append(box)

	# suppress non-maximal boxes
	for c in range(nb_class):
		sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			
			if boxes[index_i].classes[c] == 0: 
				continue
			else:
				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]
					
					if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
						boxes[index_j].classes[c] = 0
						
	# remove the boxes which are less likely than a obj_threshold
	boxes = [box for box in boxes if box.get_score() > obj_threshold]
	
	result = []
	for i in range(len(boxes)):
		if(boxes[i].classes[0]==0):
			continue
		predicted_class = "face"
		score = boxes[i].score
		result.append([predicted_class,(boxes[i].xmax+boxes[i].xmin)*img_width/2,(boxes[i].ymax+boxes[i].ymin)*img_height/2,(boxes[i].xmax-boxes[i].xmin)*img_width,(boxes[i].ymax-boxes[i].ymin)*img_height,score])

	return result

class bounding_box:
	def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		
		self.c     = c
		self.classes = classes

		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
		
		return self.label
	
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
			
		return self.score

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
	
	intersect = intersect_w * intersect_h

	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	
	union = w1*h1 + w2*h2 - intersect
	
	return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b

	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2<x3:
			return 0
		else:
			return min(x2,x4) - x3
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
	x = x - np.max(x)
	
	if np.min(x) < t:
		x = x/np.min(x)*t
		
	e_x = np.exp(x)
	
	return e_x / e_x.sum(axis, keepdims=True)

#crop
def crop(x,y,w,h,margin,img_width,img_height):
	xmin = int(x-w*margin)
	xmax = int(x+w*margin)
	ymin = int(y-h*margin)
	ymax = int(y+h*margin)
	if xmin<0:
		xmin = 0
	if ymin<0:
		ymin = 0
	if xmax>img_width:
		xmax = img_width
	if ymax>img_height:
		ymax = img_height
	return xmin,xmax,ymin,ymax

#display result
def show_results(img,results, img_width, img_height, model_age, model_gender):
	img_cp = img.copy()
	for i in range(len(results)):
		#display detected face
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2

		if(w<h):
			w=h
		else:
			h=w

		xmin,xmax,ymin,ymax=crop(x,y,w,h,1.0,img_width,img_height)

		#cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(125,125,125),2)
		#cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
		cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

		target_image=img_cp

		#analyze detected face
		xmin2,xmax2,ymin2,ymax2=crop(x,y,w,h,1.1,img_width,img_height)

		face_image = img[ymin2:ymax2, xmin2:xmax2]

		if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
			continue

		#cv2.rectangle(target_image, (xmin2,ymin2), (xmax2,ymax2), color=(0,0,255), thickness=3)

		offset=16

		lines_age=open('words/agegender_age_words.txt').readlines()
		lines_gender=open('words/agegender_gender_words.txt').readlines()
		"""lines_fer2013=open('words/emotion_words.txt').readlines()"""

		if(model_age!=None):
			shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
			img_keras = cv2.resize(face_image, (shape[1],shape[2]))
			#img_keras = img_keras[::-1, :, ::-1].copy()	#BGR to RGB
			img_keras = np.expand_dims(img_keras, axis=0)
			img_keras = img_keras / 255.0

			pred_age_keras = model_age.predict(img_keras)[0]
			prob_age_keras = np.max(pred_age_keras)
			cls_age_keras = pred_age_keras.argmax()

			age=0
			for i in range(101):
				age=age+pred_age_keras[i]*i
			label=str(int(age))

			#label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]

			cv2.putText(target_image, "Age : "+label, (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

		if(model_gender!=None):
			shape = model_gender.layers[0].get_output_at(0).get_shape().as_list()

			img_gender = cv2.resize(face_image, (shape[1],shape[2]))
			#img_gender = img_gender[::-1, :, ::-1].copy()	#BGR to RGB
			img_gender = np.expand_dims(img_gender, axis=0)
			img_gender = img_gender / 255.0

			pred_gender_keras = model_gender.predict(img_gender)[0]
			prob_gender_keras = np.max(pred_gender_keras)
			cls_gender_keras = pred_gender_keras.argmax()
			cv2.putText(target_image, "Gender : %.2f" % prob_gender_keras + " " + lines_gender[cls_gender_keras], (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
			offset=offset+16

	cv2.imshow('YoloKerasFaceDetection',img_cp)
	
MODEL_ROOT_PATH="./pretrain/"

#Load Model
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
#model_age = load_model(MODEL_ROOT_PATH+'agegender_age_squeezenet.hdf5')
model_age = load_model(MODEL_ROOT_PATH+'agegender_age101_squeezenet.hdf5')
model_gender = load_model(MODEL_ROOT_PATH+'agegender_gender_squeezenet.hdf5')


def main(argv):
	
	#Prepare WebCamera
	cap = cv2.VideoCapture(args.video if args.video else 0)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

	if args.image is not None and args.image != "":
		#Detection
		while True:
			#Face Detection
			if args.image == None:
				break
			img = args.image
			image = cv2.imread(img)
			print(image.dtype)
			image = cv2.resize(image, (1280,720))
			print(image.shape)

			#frame = cv2.imread("images/dress3.jpg")
			if (image is not None):
				img=image
				img = img[...,::-1]  #BGR 2 RGB
				inputs = img.copy() / 255.0
				img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				img_camera = cv2.resize(inputs, (416,416))
				img_camera = np.expand_dims(img_camera, axis=0)
				out2 = model_face.predict(img_camera)[0]
				results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
			else:
				print("No frame found")

			with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
				bg_image = None
			
				# Flip the image horizontally for a later selfie-view display, and convert
				# the BGR image to RGB.
				image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
				# To improve performance, optionally mark the image as not writeable to
				# pass by reference.
				image.flags.writeable = False
				results2 = selfie_segmentation.process(image)

				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				# Draw selfie segmentation on the background image.
				# To improve segmentation around boundaries, consider applying a joint
				# bilateral filter to "results.segmentation_mask" with "image".
				condition = np.stack(
					(results2.segmentation_mask,) * 3, axis=-1) > 0.1
				# The background can be customized.
				#   a) Load an image (with the same width and height of the input image) to
				#      be the background, e.g., 
				bg_image = cv2.imread('bg2.jpg')
				bg_image = cv2.resize(bg_image, (1280, 720))
				#   b) Blur the input image by applying image filtering, e.g.,
				#   bg_image = cv2.GaussianBlur(image,(55,55),0)

				if bg_image is None:
					bg_image = np.zeros(image.shape, dtype=np.uint8)
					bg_image[:] = BG_COLOR
				output_image = np.where(condition, image, bg_image)

				"""cv2.imshow('MediaPipe Selfie Segmentation', output_image)"""
				#Age and Gender Detection
			show_results(output_image ,results , img.shape[1], img.shape[0], model_age, model_gender)
			print("results")
			if cv2.waitKey(5) & 0xFF == 27:
				break
		cap.release()
		cv2.destroyAllWindows()

		
	else: #Detection
		while True:
			#Face Detection
			ret, frame = cap.read() #BGR

			#frame = cv2.imread("images/dress3.jpg")
			if (frame is not None):
				img=frame
				img = img[...,::-1]  #BGR 2 RGB
				inputs = img.copy() / 255.0
				img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				img_camera = cv2.resize(inputs, (416,416))
				img_camera = np.expand_dims(img_camera, axis=0)
				out2 = model_face.predict(img_camera)[0]
				results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
			else:
				print("No frame found")

			with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
				bg_image = None
				while cap.isOpened():
					success, image = cap.read()
					if not success:
					# If loading a video, use 'break' instead of 'continue'.
						continue

					# Flip the image horizontally for a later selfie-view display, and convert
					# the BGR image to RGB.
					image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
					# To improve performance, optionally mark the image as not writeable to
					# pass by reference.
					image.flags.writeable = False
					results2 = selfie_segmentation.process(image)

					image.flags.writeable = True
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

					# Draw selfie segmentation on the background image.
					# To improve segmentation around boundaries, consider applying a joint
					# bilateral filter to "results.segmentation_mask" with "image".
					condition = np.stack(
						(results2.segmentation_mask,) * 3, axis=-1) > 0.1
					# The background can be customized.
					#   a) Load an image (with the same width and height of the input image) to
					#      be the background, e.g., 
					bg_image = cv2.imread('bg2.jpg')
					bg_image = cv2.resize(bg_image, (1280, 720))
					#   b) Blur the input image by applying image filtering, e.g.,
					#   bg_image = cv2.GaussianBlur(image,(55,55),0)

					if bg_image is None:
						bg_image = np.zeros(image.shape, dtype=np.uint8)
						bg_image[:] = BG_COLOR
					output_image = np.where(condition, image, bg_image)

					"""cv2.imshow('MediaPipe Selfie Segmentation', output_image)"""
					#Age and Gender Detection
					show_results(output_image ,results , img.shape[1], img.shape[0], model_age, model_gender)
					if cv2.waitKey(5) & 0xFF == 27:
						break
				cap.release()
				cv2.destroyAllWindows()

if __name__=='__main__':
	main(sys.argv[1:])



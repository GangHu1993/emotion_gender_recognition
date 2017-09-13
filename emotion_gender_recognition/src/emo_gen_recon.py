import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
#image_path = sys.argv[1]
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]


class Emo_gen_recon:
	def __init__(self):
		print ('init_____________________')
		self.gender = []
		self.emotion = []
		self.res = []

	def det_emo_gen_reco(self,frame):
		self.res = []
		self.emotion = []
		self.gender = []
		bgr_image = frame
		gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		bounding_boxes = detect_faces(face_detection, gray_image)

		for face_coordinates in bounding_boxes:
			emo_gen = []
			res_dic = {}
			x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
			rgb_face = rgb_image[y1:y2, x1:x2]

			x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]
			try:
				rgb_face = cv2.resize(rgb_face, (gender_target_size))
				gray_face = cv2.resize(gray_face, (emotion_target_size))
			except:
				continue
			gray_face = preprocess_input(gray_face, False)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
			emotion_text = emotion_labels[emotion_label_arg]
			self.emotion.append(emotion_text)
			emo_gen.append(emotion_text)

			rgb_face = np.expand_dims(rgb_face, 0)
			rgb_face = preprocess_input(rgb_face, False)
			gender_prediction = gender_classifier.predict(rgb_face)
			gender_label_arg = np.argmax(gender_prediction)
			gender_text = gender_labels[gender_label_arg]
			self.gender.append(gender_text)
			emo_gen.append(gender_text)
			#print (face_coordinates)
			#res_dic[emotion_text] = face_coordinates
			res_dic[emotion_text+','+gender_text] = (int(x1),int(y1),int(x2),int(y2))
			#res_dic[emotion_text+','+gender_text] = (int(face_coordinates[0]),int(face_coordinates[1]),int(face_coordinates[2]),int(face_coordinates[3]))
			self.res.append(res_dic)
		print (self.res)
		print (type(self.res))

		return self.res


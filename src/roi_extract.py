"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

import cv2
import imutils
import numpy as np
import mediapipe as mp
from imutils import face_utils


from src.utils import *


# Different ways for ROI Extract
def haarcascade(img, faceCascade):
	"""
	desc: one way to ROI choice, crop image to have face (includes as well all non-skin parts)

	args: 
		- img::[np.array<2D>]
			image that we receive to find the face
		- faceCascade::[cascade object]
			object used to detect the face
	ret:
		- frame::[np.array<1D>]
			coords of bbox
			
		or 

		None
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)


	# Select ROI
	if len(face_rects) > 0:
		for (x, y, w, h) in face_rects:
			roi_frame = img[y:y + h, x:x + w]
		if roi_frame.size != img.size:
			roi_frame = cv2.resize(roi_frame, (500, 500))
			frame = np.ndarray(shape=roi_frame.shape, dtype="float")
			frame[:] = roi_frame * (1. / 255)
		
		return frame

	return None

# # ROI selections
# def ROI_choice(img,detector,predictor):
# 	"""
# 	desc: From full image obtain region of interest namely the face
	
# 	args:
# 		- img::[array<array<int>>]
# 		- detector::[special class]
# 			defined globals
# 		- predictor::[special class]
# 			defined globals
# 	ret:
# 		- shape::[array<2d>]
# 			contour (limited polygone) of face (e.g 68 landmarks position)
	
# 	"""
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	rects = detector(gray, 1)

# 	rect = rects[0]
# 	shape_dlib = predictor(gray, rect)
# 	shape = face_utils.shape_to_np(shape_dlib)
# 	return shape

def ROI_choice2(img, faceMesh):
	landmarks = [] 

	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceMesh.process(imgRGB)
	if results.multi_face_landmarks:
		for faceLms in results.multi_face_landmarks:				
			for id,lm in enumerate(faceLms.landmark):
				ih, iw, ic = img.shape
				x,y = int(lm.x*iw), int(lm.y*ih)

				landmarks.append([x,y])
	
	return np.asarray(landmarks)
	
# https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py
def skin_seg(img, mask='full'):
	"""
	desc: segment skin pixels from roi
	
	args:
		- img::[array<array<int>>]
	
	ret:
		- masked::[array<array<int>>]
	"""
	
	lower_hsv = np.array((0, 48, 0), dtype = "uint8") 
	upper_hsv = np.array((230,255,255), dtype = "uint8")
	lower_ycr = np.array((0, 130, 70), dtype = 'uint8')
	upper_ycr = np.array((255,180,135), dtype = 'uint8')

		
	img_hsv = color_mapping(img,'hsv')
	HSV_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
	HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

	img_YCrCb = color_mapping(img,'ycrcb')
	
	YCrCb_mask = cv2.inRange(img_YCrCb, lower_ycr, upper_ycr) 
	YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
	
	global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
	global_mask=cv2.medianBlur(global_mask,3)
	global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

	if mask == 'full':
		masked = cv2.bitwise_and(img, img, mask=global_mask)
	
	elif mask == 'hsv':
		masked = cv2.bitwise_and(img_hsv, img_hsv, mask=HSV_mask)
		print('hsv')
	else:
		masked = cv2.bitwise_and(img_YCrCb, img_YCrCb, mask=YCrCb_mask)
	return masked

def ROI_refine(img,shape,landmark,skin_flag=True, mask='full'):
	"""
	desc: refine a landmark created shape into more interesting ROIs
	
	args: 
		- img[array<array<int>>]
		- shape[array<2darray>]
			just array of points to know where we delimit the face
		- landmark[array<2darray>]
			landmarks that we chose within shape to crop the ROI
		- skin_flag[bool]
			do skin segmentation or not
		- mask[str]
			when doing skin_seg, do we use YCRcb and HSV or just HSV as a interval seg
	ret:
		- roi[array<array<int>>]
			cleaned roi that we can use immediatly for color and rppg extract
	
	"""
	roi_coord = shape[landmark]
	x,y,w,h = recthull(roi_coord)
	roi = img[y:y+h,x:x+w]
	
	if skin_flag:
		refined = skin_seg(roi, mask=mask)
		return refined
	else:
		return roi
import torch
import cv2
import math
import tensorflow as tf
import numpy as np
import glob
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig
# import flask

# APP = flask.Flask(__name__)

def perform(path, model, hitnet_depth):
	left_path = "/content/drive/MyDrive/hackathon-salesforce/images/left/"+path+".jpg"
	right_path = "/content/drive/MyDrive/hackathon-salesforce/images/right/"+path+".jpg"
	depth_path = "/content/drive/MyDrive/hackathon-salesforce/images/depth/"+path+".png"
	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/256

	# Estimate the depth
	disparity_map = hitnet_depth(left_img, right_img)
	depth_map = hitnet_depth.get_depth()

	color_disparity = draw_disparity(disparity_map)
	color_depth = draw_depth(depth_map, max_distance)
	color_real_depth = draw_depth(depth_img, max_distance)
	# cobined_image = np.hstack((left_img,color_real_depth, color_depth))
	# cv2.imshow("Estimated depth", cobined_image)
 
	# cv2.imwrite("./images/est.png", color_depth)
	# print(depth_map.shape, color_depth.shape)

	cv2.imwrite(f"./images/est_{path}.png", color_depth)

	# Inference
	results = model(left_img)
	results.save()

	c_height = left_img.shape[0]
	c_width = left_img.shape[1]/2

	for i in range(results.pandas().xyxy[0].shape[0]):
		xmin = int(results.pandas().xyxy[0].iloc[i].xmin)
		ymin = int(results.pandas().xyxy[0].iloc[i].ymin)
		xmax = int(results.pandas().xyxy[0].iloc[i].xmax)
		ymax = int(results.pandas().xyxy[0].iloc[i].ymax)
		# print(int(results.pandas().xyxy[0].iloc[i].xmin), int(results.pandas().xyxy[0].iloc[i].ymin))
		dist = 0
		cnt = 0

		cordy = (xmax-xmin)/2 + xmin
		cordx = (ymax-ymin)/2 + ymin

		ang = math.degrees(math.atan(abs(c_width - cordy)/abs(c_height-cordx)))

		clock = ""

		if ang >= 0 and ang <= 15:
			if cordy > c_width:
				clock = "12"
			else:
				clock = "12"
		elif ang >= 15 and ang <= 45:
			if cordy > c_width:
				clock = "1"
			else:
				clock = "11"
		elif ang >= 45 and ang <= 75:
			if cordy > c_width:
				clock = "2"
			else:
				clock = "10"
		else:
			if cordy > c_width:
				clock = "3"
			else:
				clock = "9"

		for x in range(xmin, xmax):
			for y in range(ymin, ymax):
				# least = min(least, depth_map[y][x])
				dist += depth_map[y][x]
				cnt += 1
		dist /= cnt
		print(f"A {results.pandas().xyxy[0].iloc[i]['name']} is {round(dist)}m at your {clock}`o clock {round(ang)}. deg")

if __name__ == "__main__":

	# APP.run()
	# Model
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

	# Select model type
	model_type = ModelType.middlebury
	# model_type = ModelType.flyingthings
	# model_type = ModelType.eth3d

	if model_type == ModelType.middlebury:
		model_path = "models/middlebury_d400.pb"
	elif model_type == ModelType.flyingthings:
		model_path = "models/flyingthings_finalpass_xl.pb"
	elif model_type == ModelType.eth3d:
		model_path = "models/eth3d.pb"

	camera_config = CameraConfig(0.546, 1000)
	max_distance = 400

	# Initialize model
	hitnet_depth = HitNet(model_path, model_type, camera_config)
	
	# Images
	paths = [
		'2',
		'1'
		] 

	for path in paths:
		print(path)
		perform(path, model, hitnet_depth)
from tracker import *
from ctypes import *
from calc_xyz_all import *
from stitch_img import *
from datetime import datetime
import numpy as np
import subprocess
import random
import os
import math
import cv2
import time
import darknet
import argparse
import serial
from threading import Thread, enumerate
from queue import Queue
import queue
import pygame
from calibration_store import load_stereo_coefficients
import json
import EasyPySpin
from PIL import Image
from itertools import product


#Read from Arduino
#try:
	#arduino = serial.Serial('/dev/ttyACM1', timeout=1, baudrate=9600)
#except:
	#print('Please check the port')

stereo_cam_yml = './stereo_cam.yml'
_channel = '101'
tracker_queue_size = 64
def parser():
	parser = argparse.ArgumentParser(description="YOLO Object Detection")
	parser.add_argument("--input1", type=str, default="./Drone_Test_1.mp4", help="video source. If empty, uses webcam 0 stream")
	#parser.add_argument("--input2", type=str, default="data/V_DRONE_001.mp4"#stearm 2,
						#help="video source. If empty, uses webcam 0 stream")
	parser.add_argument("--out_filename1", type=str, default="Drone_Test_Out.mp4",
						help="inference video name. Not saved if empty")
	parser.add_argument("--out_filename2", type=str, default="",
						help="inference video name. Not saved if empty")#created 2 instances for this
	parser.add_argument("--weights", default="yolov4.weights",
						help="yolo weights path")
	parser.add_argument("--dont_show1", action='store_true',
						help="windown inference display. For headless systems")#created 2 instances for this
	parser.add_argument("--dont_show2", action='store_true',
						help="windown inference display. For headless systems")
	parser.add_argument("--ext_output1", action='store_true',
						help="display bbox coordinates of detected objects") #created 2 instances for this
	parser.add_argument("--ext_output2", action='store_true',
						help="display bbox coordinates of detected objects")
	parser.add_argument("--config_file", default="./cfg/yolov4-custom.cfg",
						help="path to config file")
	parser.add_argument("--data_file", default="./cfg/drone.data",
						help="path to data file")
	parser.add_argument("--thresh", type=float, default=.7,
						help="remove detections with confidence below this value")
	return parser.parse_args()
	


def convert2relative(bbox):
	"""
	YOLO format use relative coordinates for annotation
	"""
	x, y, w, h  = bbox
	_height     = darknet_height
	_width      = darknet_width
	return x/_width, y/_height, w/_width, h/_height
		

def convert2original(size, bbox):
	"""
	Convert YOLO annotations to pixels in original image
	"""
	x, y, w, h = convert2relative(bbox)

	image_w, image_h = size

	orig_x       = int(x * image_w)
	orig_y       = int(y * image_h)
	orig_width   = int(w * image_w)
	orig_height  = int(h * image_h)

	bbox_converted = [orig_x, orig_y, orig_width, orig_height]

	return bbox_converted
	
def calc_rot_matrix(roll, pitch, yaw):

	R_mtx = [[math.cos(yaw)*math.cos(pitch), math.cos(pitch)*math.sin(yaw), -math.sin(pitch)],
				[math.sin(roll)*math.sin(pitch)*math.cos(yaw)- math.sin(yaw)*math.cos(roll), math.cos(yaw)*math.cos(roll)+math.sin(yaw)*math.sin(pitch)*math.sin(roll), math.cos(pitch)*math.sin(roll)],
				[math.sin(roll)*math.sin(yaw) + math.cos(roll)*math.sin(pitch)*math.cos(yaw), -math.cos(yaw)*math.sin(roll) + math.cos(roll)*math.sin(pitch)*math.sin(yaw) ,   math.cos(pitch)*math.cos(roll)]]

	return R_mtx


def video_process(frame_queue1, frame_queue2, darknet_image_queue1, darknet_image_queue2):
	#init rectification
	#newCamMtx, roi = cv2.getOptimalNewCameraMatrix(K,D, (width, height), 1, (width, height))
	#leftMapX, leftMapY = cv2.initUndistortRectifyMap(K, D, None, newCamMtx, (width, height), cv2.CV_32FC1)
	
	while cap1.isOpened() and cap2.isOpened(): #while cap1.isOpened() or cap2.isOpened():
	   
		#raw_frame = ffmpeg_process.stdout.read(width*height*3)
		ret1, frame1 = cap1.read()
		ret2, frame2 = cap2.read()
		if not ret1 and ret2:
			break
		#print(f"Initial shape of the frame:: {frame1.shape}")
        	#if not ret1 and ret2:
            		#break
		#if len(raw_frame) != 0:    
		#frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

		#perform rectification
		#frame = cv2.remap(frame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

		#frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BayerBG2BGR)
		#frame_resized1 = cv2.resize(frame_rgb1, (darknet_width, darknet_height),
								   #interpolation=cv2.INTER_LINEAR)
								   
		frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BayerBG2BGR)#cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
		frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BayerBG2BGR)#cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
		frame_resized1 = cv2.resize(frame_rgb1, (darknet_width, darknet_height),
					interpolation=cv2.INTER_LINEAR)
		frame_resized2 = cv2.resize(frame_rgb2, (darknet_width, darknet_height),
					interpolation=cv2.INTER_LINEAR)                
		
		#print(f"resized frame shape:: {frame_resized1.shape}")
		
		if not frame_queue1.empty():
			try:
				frame_queue1.get_nowait()   
			except queue.Empty:
				pass
				
		if not frame_queue2.empty():
			try:
				frame_queue2.get_nowait()   
			except queue.Empty:
				pass

		frame_queue1.put(frame_rgb1) #frame_resized1
		frame_queue2.put(frame_rgb2) #frame_resized2
		    
		darknet_image_queue1.put(frame_rgb1)
		darknet_image_queue2.put(frame_rgb2)
		
		#global stop_threads
		if cv2.waitKey(30) == ord("q"):
			#stop_threads = True
		#if stop_threads == True:
			break
		#cv2.imshow("cam1", frame)

	#ffmpeg_process1.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process2.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process1.wait()  # Wait for FFmpeg sub-process to finish
	#ffmpeg_process2.wait()  # Wait for FFmpeg sub-process to finish
	cap1.release()
	cap2.release()
	cv2.destroyAllWindows()

def tile(image):
    list_1 = []
    #img = Image.fromarray(image)
    h, w, _ = image.shape
    #w, h = img.size
    d_h = int(h/2)
    d_w = int(w/2)
    grid = product(range(0, h-h%d_h, d_h), range(0, w-w%d_w, d_w))
    for i, j in grid:
        print(j,i)
        box = (j, i, j+d_w, i+d_h)
        #out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        list_1.append(img.crop(box))
        list_1 = np.array(list_1)
        print(len(list_1))
        
        #img.crop(box).save(out)
    return np.array[list_1]

def inference(darknet_image_queue1,darknet_image_queue2, detections_queue1, detections_queue2):
	while cap1.isOpened() or cap2.isOpened():
		
		
		
		darknet_image1 = darknet_image_queue1.get()
		darknet_image2 = darknet_image_queue2.get()
		#print(f"shape of darknet_image1:: {darknet_image1}")
		
		#darknet_image3 = cv2.resize(darknet_image1, (640, 480))
		cv2.imshow('darknet_image1',darknet_image3)
		if cv2.waitKey(1) == ord("q"):
			break
		
		crop_1 = tile(darknet_image1)
		crop_2 = tile(darknet_image2)
		#image = np.array(crop_1)
		#print(f"shape of crop_1:: {len(crop_1)}")
		
		#crop_1 = [crop_1[1]
		#crop_2 = [crop_2[1]]
		#print(len(crop_1))
		
		detect_1 = []
		detect_2 = []
		
		#cv2.namedWindow('c1', cv2.WINDOW_NORMAL)
		
		for c1,c2 in zip(crop_1, crop_2):
			
									
			img_for_detect1 = darknet.make_image(darknet_width, darknet_height, 3)
			img_for_detect2 = darknet.make_image(darknet_width, darknet_height, 3)
			
			
			cv2.imshow('c1',np.array(c1))
			if cv2.waitKey(1) == ord("q"):
				break
			
			
			darknet.copy_image_from_bytes(img_for_detect1, np.array(c1).tobytes())
			darknet.copy_image_from_bytes(img_for_detect2, np.array(c2).tobytes())
			
			#img_x = darknet.copy_image_from_bytes(img_for_detect1, np.array(c1).tobytes())
			#print(f"image_x type is:: {type(img_for_detect1)}")
			#print(f"c1 type is:: {type(np.array(c1).tobytes())}")
			
			
			detections1 = darknet.detect_image(network, class_names, img_for_detect1, thresh=args.thresh) #img_for_detect1
			detections2 = darknet.detect_image(network, class_names, img_for_detect2, thresh=args.thresh) #img_for_detect2
			
			detect_1.append(detections1)
			print(f"Detections1:: {detections1}")
			detect_2.append(detections2)
			
			darknet.free_image(img_for_detect1)
			darknet.free_image(img_for_detect2)
			
		
			
		# stitch the image here
		detections_1 = stitch_images(crop_1, detect_1,2,2)
		print(f"detections_1:: {detections_1}")
		detections_2 = stitch_images(crop_2, detect_2,2,2)
		
		if not detections_queue1.empty():
				try:
					detections_queue1.get_nowait()   
				except queue.Empty:
					pass
		if not detections_queue2.empty():
				try:
					detections_queue2.get_nowait()   
				except queue.Empty:
					pass
		detections_queue1.put(detections_1)
		detections_queue2.put(detections_2)
		#darknet.free_image(darknet_image1)
		#darknet.free_image(darknet_image2)
		#global stop_threads
		#if stop_threads == True:
            		#break
	if cap1.isOpened():
		cap1.release()
	if cap2.isOpened():
		cap2.release()

	#ffmpeg_process1.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process2.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process1.wait()  # Wait for FFmpeg sub-process to finish
	#ffmpeg_process2.wait()  # Wait for FFmpeg sub-process to finish

	cv2.destroyAllWindows()

def tracking(detections_queue1, detections_queue2, object_queue1, object_queue2):
	'''
	detection_queues contain tuples [(label, score, [x,y,w,h]), (str, float, (float, float, float, float))]
	object_queues contain tuples [((x,y), label, score, name), ((float,float), str, float, int) ...]
	'''
	frame_width = 608#1920
	frame_height = 608#1080
	show_suppressed = False
	size = (frame_width, frame_height)
	system1 = KFSystem(nms_threshold = 0.5, frame_width = frame_width, frame_height = frame_height, score_threshold = 0.9, tolerance = 30, show_suppressed = show_suppressed)
	system2 = KFSystem(nms_threshold = 0.5, frame_width = frame_width, frame_height = frame_height, score_threshold = 0.9, tolerance = 30, show_suppressed = show_suppressed)
	oldTick, dt = time.time(), 0
	while cap1.isOpened() and cap2.isOpened():
		bbox1, labels1, scores1 = [], [], []
		bbox2, labels2, scores2 = [], [], []
		detections1 = detections_queue1.get()
		detections2 = detections_queue2.get()
		newTick= time.time()
		dt = newTick - oldTick
		oldTick = newTick
		#dt = 0
		# print(f'tracking:{detections1}')
		for label, score, bbox in detections1:
			bb = convert2original(size, bbox)
			#print(f"the bbox coordinates are:: {bb}")
			bbox1+=[bb]
			labels1+=[label]
			scores1+=[score]
			
		for label, score, bbox in detections2:
			bb = convert2original(size, bbox)
			bbox2+=[bb]
			labels2+=[label]
			scores2+=[score]
		system1.run(np.array(bbox1), np.array(labels1), np.array(scores1), dt)
		system2.run(np.array(bbox2), np.array(labels2), np.array(scores2), dt)
		measurements1, measurements2 = [], []
		names1, names2 = [], [] 
		if not object_queue1.empty():
				try:
					object_queue1.get_nowait()   
				except queue.Empty:
					pass
		if not object_queue2.empty():
				try:
					object_queue2.get_nowait()   
				except queue.Empty:
					pass
		if system1.trackers:
			for detection in sorted(system1.trackers[:tracker_queue_size], key= lambda x: x.measurement[0]):
				measurements1.append((detection.measurement, detection.label, detection.score, detection.name))
				names1.append(detection.name)
			
		if system2.trackers:
			for detection in sorted(system2.trackers[:tracker_queue_size], key= lambda x: x.measurement[0]):
				measurements2.append((detection.measurement, detection.label, detection.score, detection.name))
				names2.append(detection.name)
		# print(f'Detections in system 1: {names1}')
		# print(f'Detections in system 2: {names2}')
		#print(f"measurements are:: {measurements1}")	
		object_queue1.put(measurements1)
		object_queue2.put(measurements2)
			
		# print(f'names1:{names1}')
		# print(f'names2:{names2}')
	if cap1.isOpened():
		cap1.release()
	if cap2.isOpened():
		cap2.release()

def drawing(frame_queue1, frame_queue2, object_queue1, object_queue2):
	"""
	frame_queues contain frames from cameras
	object_queues contain tuples [((x,y), label, score, name), ((float,float), str, float, int) ...]
	"""
	def match_points(points1, points2):
		""" 
		minimise distance of matches in y and ensure x1 (leftcam) > x2 (rightcam)
		points1 is expected to be sorted by detection confidence descending order
		"""

		matches = []
		# print(points1, points2)
		while(points1):
			p1 = points1[0][0]
			if len(points2):
				diff = [abs(p1[1] - p2[0][1]) for p2 in points2]
				while(True):
					ind = np.argmin(diff)
					match = points2[ind][0]
					if match[0] < p1[0]:
						try:
							matches.append((p1, points2[ind][0], points1[0][2], points2[ind][2]))
							if len(points2[:ind]):
								if len(points2[ind+1:]):
									points2 = np.concatenate([points2[:ind], points2[ind+1:]])
								else:
									points2 = points2[:ind]
							else:
								points2 = points2[ind+1:]
							points1 = points1[1:]
							break
						except Exception as e:
							print(f'Exception occured in match_points:\n{e}')
					else:
						diff[ind] = 100000
						if (np.array(diff)==100000).all():
							points1 = points1[1:]
							break
			else:
				break
		return matches
	random.seed(3)  # deterministic bbox colors
	#video1 = set_saved_video(cap1, args.out_filename1, (video_width1, video_height1))
	#video2 = set_saved_video(cap2, args.out_filename2, (video_width2, video_height2))

	# choose methods to print estimated distance from
	fov_method = True
	triangulation_method = True

	width, height = 1920, 1080 #1920, 1080
	newName, left, right = 0, {}, {}
	camera_l = camera_realtimeXYZ(0)
	camera_r = camera_realtimeXYZ(1)
	fov_h = 60.0 *np.pi/(180*height)
	fov_w = 112.1*np.pi/(180*width)

	center1, center2 = camera_l.camera_center(), camera_r.camera_center()

	while cap1.isOpened() and cap2.isOpened():
		
		frame1 = frame_queue1.get()
		frame2 = frame_queue2.get()
		objects1 = object_queue1.get()
		objects2 = object_queue2.get()
		#height, width, channel = frame1.shape                       
		
		# roll_cam = 0.0
		# pitch_cam = math.pi/2.0 
		# yaw_cam = math.pi/2.0
		R_mtx_cam = calc_rot_matrix(0, 0, 0)

		with open('yaw_data.txt', 'r') as f:
			file_data = f.read(6)
			if len(file_data):
				yaw_cam = round(math.radians(float(file_data)),3)
		roll_cam = 0.0
		pitch_cam = 0.0
		R_mtx_setup = calc_rot_matrix(roll_cam, pitch_cam, yaw_cam)

		if frame1 is not None and frame2 is not None:
			bbrad1, bbrad2 = 85, 85
			uv1, uv2 = [], []
			dets1, dets2 = [], []
			for obj, label, score, name in objects1:
				#frame1 = cv2.circle(frame1, (int(obj[0]),int(obj[1])), 85, (0, 0, 255),thickness= 5, lineType=8, shift=0)
				uv1.append(([obj[0], obj[1]], score, name))
				dets1.append((str(label), score, [max(obj[0],0), max(obj[1],0), min(2*bbrad1, width-obj[0]), min(2*bbrad1, height-obj[1])]))
			cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
			

			#leftMapX1, leftMapY1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
			#left_rectified_1 = cv2.remap(image1, leftMapX1, leftMapY1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

			for obj, label, score, name in objects2:
				#frame2 = cv2.circle(frame2, (int(obj[0]),int(obj[1])), 85, (0, 0, 255),thickness= 5, lineType=8, shift=0)
				uv2.append(([obj[0], obj[1]], score, name))
				dets2.append((str(label), score, [max(obj[0],0), max(obj[1],0), min(2*bbrad2, width-obj[0]), min(2*bbrad2, height-obj[1])]))
			# print(f'drawing:{dets1}')
			frame1 = darknet.draw_boxes(dets1, frame1, class_colors)
			frame2 = darknet.draw_boxes(dets2, frame2, class_colors)
			cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
			uv1, uv2 = sorted(uv1, key = lambda x:-float(x[1])), sorted(uv2, key = lambda x:-float(x[1]))
			matches = match_points(uv1, uv2)
			if uv1 and uv2:
				used = set()
				for i in range(len(matches)):
					
					combined_label = f'{matches[i][2]}:{matches[i][3]}'
					if matches[i][2] in left and left[matches[i][2]] not in used:
						currentName = left[matches[i][2]]
						right[matches[i][3]] = currentName
						used.add(left[matches[i][2]])
					elif matches[i][3] in right and right[matches[i][3]] not in used:
						currentName = right[matches[i][3]]
						left[matches[i][2]] = currentName
						used.add(right[matches[i][3]])
					else:
						left[matches[i][2]] = newName
						right[matches[i][3]] = newName
						currentName = newName
						newName+=1
						used.add(currentName)
					xyz1_ = np.array([matches[i][0][0], matches[i][0][1]])
					xyz2_ = np.array([matches[i][1][0], matches[i][1][1]])
					frame1 = cv2.putText(frame1, f'{currentName}', (int(xyz1_[0]), int(xyz1_[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
					frame2 = cv2.putText(frame2, f'{currentName}', (int(xyz2_[0]), int(xyz2_[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
					if fov_method:
						d11, d12 = (xyz1_[0] -center1[0]) * (fov_w), (xyz1_[1]-center1[1])* (fov_h)
						x_c1, y_c1, z_c1 = np.sin(d11),0, np.cos(d11)
						xyz1 = np.array([[x_c1], [y_c1], [z_c1]])
						r1 = math.sqrt(y_c1*y_c1 + z_c1*z_c1)
						len_c1 = math.sqrt(y_c1*y_c1 + z_c1*z_c1 + x_c1*x_c1)
						d11, d12 = (xyz2_[0] -center2[0]) * (fov_w), (xyz2_[1]-center2[1])* (fov_h)
						x_c2, y_c2, z_c2 = np.sin(d11), 0, np.cos(d11)
						xyz2 = np.array([[x_c2], [y_c2], [z_c2]])
						r2 = math.sqrt(y_c2*y_c2 + z_c2*z_c2)
						len_c2 = math.sqrt(y_c2*y_c2 + z_c2*z_c2 + x_c2*x_c2)
						sep_len = 0.31 

						g = [[x_c1/len_c1, -x_c2/len_c2],[r1/len_c1, -r2/len_c2]]
						m_var = -(g[0][0]*g[0][1] + g[1][0]*g[1][1])

						mat_a = np.array([[g[0][0]**2 + g[1][0]**2, -m_var],[m_var, -(g[0][1]**2 + g[1][1]**2)]], dtype = np.float32)
						mat_b = sep_len*np.array([[g[0][0]],[-g[0][1]]], dtype = np.float32)
						alphabeta = np.linalg.pinv(mat_a).dot(mat_b)
						alpha, beta = alphabeta[0][0], alphabeta[1][0]
						print(f'dist by fov of {currentName}: {(alpha+beta)/2}')
					if triangulation_method:
						xyz1, xyz_cam1 = camera_l.calculate_XYZ(xyz1_[0],xyz1_[1],R_mtx_setup,R_mtx_cam, True)
						xyz2, xyz_cam2 = camera_r.calculate_XYZ(xyz2_[0],xyz2_[1],R_mtx_setup, R_mtx_cam, True)
						x_c1 = xyz_cam1[0][0]
						y_c1 = 0 #xyz_cam1[1][0]
						z_c1 = xyz_cam1[2][0]
						x_c2 = xyz_cam2[0][0]
						y_c2 = 0 #xyz_cam2[1][0]
						z_c2 = xyz_cam2[2][0]
						r1 = math.sqrt(y_c1*y_c1 + z_c1*z_c1)
						len_c1 = math.sqrt(y_c1*y_c1 + z_c1*z_c1 + x_c1*x_c1)
						r2 = math.sqrt(y_c2*y_c2 + z_c2*z_c2)
						len_c2 = math.sqrt(y_c2*y_c2 + z_c2*z_c2 + x_c2*x_c2)
						sep_len = 0.31 
						g = [[x_c1/len_c1, -x_c2/len_c2],[r1/len_c1, -r2/len_c2]]
						m_var = (g[0][0]*g[0][1] + g[1][0]*g[1][1])

						mat_a = np.array([[g[0][0]**2 + g[1][0]**2, m_var],[-m_var, -(g[0][1]**2 + g[1][1]**2)]], dtype = np.float32)
						mat_b = sep_len*np.array([[g[0][0]],[-g[0][1]]], dtype = np.float32)
						alphabeta = np.linalg.pinv(mat_a).dot(mat_b)
						alpha, beta = alphabeta[0][0], alphabeta[1][0]
						print(f'dist by tri of {currentName}: {(alpha+beta)/2}')
						frame1 = cv2.putText(frame1, f'{xyz1},{x_c1},{y_c1},{z_c1}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
						frame2 = cv2.putText(frame2, f'{xyz2},{x_c2},{y_c2},{z_c2}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
						frame1 = cv2.circle(frame1, (int(center1[0]),int(center1[1])), 40, (0, 0, 255),thickness= 2, lineType=8, shift=0)
						frame2 = cv2.circle(frame2, (int(center2[0]),int(center2[1])), 40, (0, 0, 255),thickness= 2, lineType=8, shift=0)
			cv2.resizeWindow('image1', 900, 600)
			cv2.resizeWindow('image2', 900, 600)               
			cv2.imshow('image1', frame1)
			cv2.imshow('image2', frame2)
			
			if cv2.waitKey(1) == ord("q"):
				break
			#global stop_threads
			#if stop_threads == True:
                    		#break

 #------------------------------------------------------------------------------------------------------------------
									
	if cap1.isOpened():
		cap1.release()
	if cap2.isOpened():
		cap2.release()
	#cv2.waitKey(0)
	#ffmpeg_process1.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process2.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
	#ffmpeg_process1.wait()  # Wait for FFmpeg sub-process to finish
	#ffmpeg_process2.wait()  # Wait for FFmpeg sub-process to finish

	
	cv2.destroyAllWindows()

 #------------------------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
	frame_queue1 = Queue()
	frame_queue2 = Queue()
	darknet_image_queue1 = Queue(maxsize=10)
	darknet_image_queue2 = Queue(maxsize=10)
	detections_queue1 = Queue(maxsize=10)
	detections_queue2 = Queue(maxsize=10)
	object_queue1 = Queue(maxsize=tracker_queue_size)
	object_queue2 = Queue(maxsize=tracker_queue_size)
	
	#K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(stereo_cam_yml)  # Get cams params
	
	args = parser()
	#check_arguments_errors(args)
	network, class_names, class_colors = darknet.load_network(
			args.config_file,
			args.data_file,
			args.weights,
			batch_size=1
		)
	#Making changes from now on
	darknet_width = darknet.network_width(network)
	darknet_height = darknet.network_height(network)
	input_path1 = "rtsp://admin:hikVision123@169.254.89.249:554/Streaming/Channels/" + _channel 
	input_path2 = "rtsp://admin:hikVision123@169.254.89.250:554/Streaming/Channels/" + _channel
	
	#command1 = ['ffmpeg', # Using absolute path for example (in Linux replacing 'C:/ffmpeg/bin/ffmpeg.exe' with 'ffmpeg' supposes to work).
		   #'-rtsp_flags', 'listen',   # The "listening" feature is not working (probably because the stream is from the web)
		   #'-rtsp_transport', 'tcp',   # Force TCP (for testing)
		   #'-max_delay', '3000000',   # 30 seconds (sometimes needed because the stream is from the web).
		   #'-hwaccel', 'cuda',             
		   #'-i', input_path1,
	#        '-c:v', 'h264_nvenc',           # h264_nvenc
		   #'-vf', 'scale=1920:1080',                   #'format=bgr24'
		   #'-preset', 'fast',
		   #'-tune', 'zerolatency',
		   #'-pix_fmt', 'bgr24',
		   #'-an', 
		   #'-crf', '23'
		   #'-f', 'rawvideo',           # Video format is raw video
	           #'-c:v', 'h264_nvenc', 
		   #'-pix_fmt', 'bgr0', #'yuvj420p'
							# bgr24 pixel format matches OpenCV default pixels format.
		   #'-async','0',
		   #'pipe:'] #'-an'
		   
	#command2 = ['ffmpeg', # Using absolute path for example (in Linux replacing 'C:/ffmpeg/bin/ffmpeg.exe' with 'ffmpeg' supposes to work).
		   #'-rtsp_flags', 'listen',   # The "listening" feature is not working (probably because the stream is from the web)
		   #'-rtsp_transport', 'tcp',   # Force TCP (for testing)
		   #'-max_delay', '3000000',   # 30 seconds (sometimes needed because the stream is from the web).
		   #'-hwaccel', 'cuda',             
		   #'-i', input_path2,
	#        '-c:v', 'h264_nvenc',           # h264_nvenc
		   #'-vf', 'scale=1920:1080',
		   #'-pix_fmt', 'bgr24',
		   #'-an',  
		   #'-preset', 'fast',
		   #'-tune', 'zerolatency',
		   #'-crf', '23'
		   #'-f', 'rawvideo',           # Video format is raw video
	           #'-c:v', 'h264_nvenc', 
		   #'-pix_fmt', 'bgr0',
									# bgr24 pixel format matches OpenCV default pixels format.
		   #'-async','0',
		   #'pipe:']

	# Open sub-process that gets in_stream as input and uses stdout as an output PIPE.
	#ffmpeg_process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, bufsize = 10**5) #1**8
	#ffmpeg_process2 = subprocess.Popen(command2, stdout=subprocess.PIPE, bufsize = 10**5)
	
	#cap1 = cv2.VideoCapture(input_path1)
	#cap2 = cv2.VideoCapture(input_path2)
	
	cap1 = EasyPySpin.VideoCapture(0)
	cap2 = EasyPySpin.VideoCapture(0)
	
	framerate1 = cap1.get(5) #frame rate
	framerate2 = cap2.get(5) #frame rate
	
	
	# Get resolution of input video
	width1  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
	height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# width1, width2 = 1920,1920
	# height1, height2 = 1080,1080
	# Get resolution of input video
	width2  = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
	height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
		
	
	#Thread(target=video_capture, args=(width1, height1, width2, height2, ffmpeg_process1, ffmpeg_process2, frame_queue1,frame_queue2, darknet_image_queue1,darknet_image_queue2)).start()
	
	Thread(target=video_process, args=(frame_queue1, frame_queue2, darknet_image_queue1, darknet_image_queue2)).start()
	#Thread(target=video_process, args=(frame_queue2, darknet_image_queue2,camera_realtimeXYZ(1))).start()
	Thread(target=inference, args=(darknet_image_queue1,darknet_image_queue2, detections_queue1, detections_queue2)).start()
	Thread(target=tracking, args=(detections_queue1,detections_queue2, object_queue1, object_queue2)).start()
	Thread(target=drawing, args=(frame_queue1, frame_queue2, object_queue1, object_queue2)).start()

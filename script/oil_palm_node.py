#!/usr/bin/env python

#Title: Python Subscriber for loosefruit recognition
#Author: Nurshafikah Darwis ---[6/8/2020]
#Description: loosefruit_oil_palm_detection(node) python

from __future__ import print_function
from __future__ import division

# import the necessary packages
from collections import deque
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import tensorflow as tf
import imutils
import time
import cv2
import os
import rospkg
import sys
import rospy
import numpy as np

# import the necessary ROS messages 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


from sensor_msgs.msg import RegionOfInterest

from common_tracking_application.objcenter import objCenter
from common_face_application.msg import objCenter as objCoord

class oil_palm_node:

	def __init__(self,buffer=16):

		rospy.logwarn("deep learning (ROI) node [ONLINE]")

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()
		self.roi = RegionOfInterest()
		self.objectCoord = objCoord()

		# define the lower and upper boundaries of the "red"
		# fruit in the HSV color space, then initialize the
		# list of tracked points
		self.lower_red = (0, 120, 70)
        	self.upper_red = (10, 255, 255)
        	self.lowerRed = (170, 120, 70)
        	self.upperRed = (180, 255, 255)

		self.pts = deque(maxlen=buffer)
		self.buffer = buffer

		self.image_recieved = False

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)
		
		# import lenet files
		self.p = os.path.sep.join([self.rospack.get_path('loosefruit_classification')])
		self.libraryDir = os.path.join(self.p, "model")
		self.lenet_filename = self.libraryDir + "/lenet_sawit13.hdf5"
		self.model = load_model(self.lenet_filename)

		# Subscribe to Image msg
		image_topic = "/cv_camera/image_raw"
		self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

		# Subscribe to CameraInfo msg
		cameraInfo_topic = "/cv_camera/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo,
			self.cbCameraInfo)

		#Publish to RegionOfInterest msg
		roi_topic = "/fruitROI_robot1"
		self.roi_pub = rospy.Publisher(roi_topic, RegionOfInterest, queue_size=10)

			
		# Publish to objCenter msg
		objCoord_topic = "/objCoord_robot1"
		self.objCoord_pub = rospy.Publisher(objCoord_topic, objCoord, queue_size=10)


		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
			self.cv_image = cv2.flip(self.cv_image, 1)

		except CvBridgeError as e:
			print(e)

		if self.cv_image is not None:

			self.image_received = True
		else:
			self.image_received = False

	# Get CameraInfo
	def cbCameraInfo(self, msg):

		self.imgWidth = msg.width
		self.imgHeight = msg.height

		# calculate the center of the frame as this is where we will
		# try to keep the object

		self.centerX = self.imgWidth // 2
		self.centerY = self.imgHeight // 2

	# Show the output frame
	def cbShowImage(self):

		cv2.imshow("loose fruit detector (ROI)", self.cv_image)
		cv2.waitKey(1)

	
	# Detect the fruit(s)
	def cbFruit(self):
		if self.image_received:

			# resize the frame, blur it, and convert it to the HSV
			# color space
			frame = imutils.resize(self.cv_image, width=self.imgWidth)
			# blurred = cv2.GaussianBlur(self.cv_image, (11, 11), 0)
			gray= cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
			hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

			# construct a mask for the color "green", then perform
			# a series of dilations and erosions to remove any small
			# blobs left in the mask
			mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
			mask2 = cv2.inRange(hsv, self.lowerRed, self.upperRed)
			mask = mask1 + mask2
			mask = cv2.erode(mask, None, iterations=1)
			mask = cv2.dilate(mask, None, iterations=1)

			# find contours in the mask and initialize the current
			# (x, y) center of the ball
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			center = None
			
			# only proceed if at least one contour was found
			objectLoc = objCenter(cnts, (self.centerX, self.centerY))
			((self.objX, self.objY), rect) = objectLoc


#			self.pubObjCoord()

			# only proceed if at least one contour was found
			if len(cnts) > 0:
				# find the largest contour in the mask, then use
				# it to compute the minimum enclosing circle and
				# centroid
				c = max(cnts, key=cv2.contourArea)
				((cX, cY), radius) = cv2.minEnclosingCircle(c)
				(self.x, self.y, self.w, self.h) = cv2.boundingRect(c)
		
 				roi = gray[self.y:self.y + self.h, self.x:self.x + self.w]
        			roi = cv2.resize(roi,(28,28))
         			roi = img_to_array(roi)
	 			roi = roi.astype("float") / 255.0
         			roi = np.expand_dims(roi, axis=0)
				
				(fruit, notFruit) = self.model.predict(roi)[0]
#				rospy.loginfo(self.model.predict(roi)[0][1])
         			label = "Fruit" if (fruit > notFruit) and fruit > 0.8 else "Not fruit"

					
				# only proceed if the radius meets a minimum size
				if radius > 10:
					# draw the circle and centroid on the frame,
					# then update the list of tracked points
					cv2.rectangle(self.cv_image, (int(self.x), int(self.y)), 
						(int(self.x)+int(self.w),int(self.y)+int(self.h)),(255, 255, 255), 2)
					cv2.putText(self.cv_image, label, (int(self.x),int(self.y)-10),	
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
#					cv2.putText(self.cv_image, "%.2f" % (self.model.predict(roi)[0][0]), 
#						(int(self.x),int(self.y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
					
				if label == "Fruit" :
#					self.pubRegionofInterest()
					self.pubObjCoord()


			self.cbShowImage()

			# Allow up to one second to connection
			rospy.sleep(0.01)
		else:
			rospy.logerr("No images recieved")

#	 Publish to RegionOfInterest msg
	def pubRegionofInterest(self):

		self.roi.x_offset = self.x
		self.roi.y_offset = self.y
		self.roi.width = self.x + self.w
		self.roi.height = self.y + self.h

		self.roi_pub.publish(self.roi)
#	
	def pubObjCoord(self):

		self.objectCoord.centerX = self.objX
		self.objectCoord.centerY = self.objY

		self.objCoord_pub.publish(self.objectCoord)


	# rospy shutdown callback
	def cbShutdown(self):
		try:
			rospy.logwarn("oil palm (ROI) node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('oil_palm_node', anonymous=False)
	oil_palm = oil_palm_node()

	# Camera preview
	while not rospy.is_shutdown():
		oil_palm.cbFruit()

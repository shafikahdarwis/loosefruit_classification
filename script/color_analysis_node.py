#!/usr/bin/env python

#Title: Python Subscriber for Tank Navigation
#Author: Khairul Izwan Bin Kamsani - [23-01-2020]
#Description: Tank Navigation Subcriber Nodes (Python)

from __future__ import print_function
from __future__ import division

# import the necessary packages
from imutils import face_utils
from collections import deque
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

class color_analysis_node:

	def __init__(self, buffer=16):

		rospy.logwarn("Color analysis (ROI) node [ONLINE]")

		self.bridge = CvBridge()

		# define the lower and upper boundaries of the "green"
		# ball in the HSV color space, then initialize the
		# list of tracked points
		self.lower_red = (0, 120, 70)
        	self.upper_red = (10, 255, 255)
        	self.lowerRed = (170, 120, 70)
        	self.upperRed = (180, 255, 255)
        	
        	self.counter = 0

		self.pts = deque(maxlen=buffer)
		self.buffer = buffer

		self.image_recieved = False

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# Subscribe to Image msg
		image_topic = "/cv_camera/image_raw"
		self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

		# Subscribe to CameraInfo msg
		cameraInfo_topic = "/cv_camera/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo,
			self.cbCameraInfo)

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

	# Image information callback
	def cbInfo(self):

		print('center of fruit:{},{}'.format(self.dX,self.dY)) 
		cv2.putText(self.cv_image, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
		cv2.putText(self.cv_image, "dx: {}, dy: {}".format(self.dX, self.dY), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 1)
		# fontFace = cv2.FONT_HERSHEY_DUPLEX
		# fontScale = 0.5
		# color = (255, 255, 255)
		# thickness = 1
		# lineType = cv2.LINE_AA
		# bottomLeftOrigin = False # if True (text upside down)
		#
		# self.timestr = time.strftime("%Y%m%d-%H:%M:%S")
		#
		# cv2.putText(self.cv_image, "{}".format(self.timestr), (10, 20),
		# 	fontFace, fontScale, color, thickness, lineType,
		# 	bottomLeftOrigin)
		# cv2.putText(self.cv_image, "Sample", (10, self.imgHeight-10),
		# 	fontFace, fontScale, color, thickness, lineType,
		# 	bottomLeftOrigin)
		# cv2.putText(self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight),
		# 	(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale,
		# 	color, thickness, lineType, bottomLeftOrigin)

	# Detect the face(s)
	def cbFace(self):
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
			if len(cnts) > 0:
				# find the largest contour in the mask, then use
				# it to compute the minimum enclosing circle and
				# centroid
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

				# only proceed if the radius meets a minimum size
				if radius > 10:
					# draw the circle and centroid on the frame,
					# then update the list of tracked points
					cv2.circle(self.cv_image, (int(x), int(y)), int(radius),
						(0, 255, 255), 2)
					cv2.circle(self.cv_image, center, 5, (0, 0, 255), -1)
					# update the points queue
					self.pts.appendleft(center)

			# loop over the set of tracked points
			for i in range(1, len(self.pts)):
				# if either of the tracked points are None, ignore
				# them
				if self.pts[i - 1] is None or self.pts[i] is None:
					continue

				if self.counter >= 10 and i == 1 and self.pts[10] is not None:
		  			# compute the difference between the x and y
		  			# coordinates and re-initialize the direction
		  			# text variables
		  			self.dX = self.pts[-10][0] - self.pts[i][0]
		  			self.dY = self.pts[-10][1] - self.pts[i][1]
		  			(self.dirX, self.dirY) = ("", "")

		  			# ensure there is significant movement in the
		  			# x-direction
		  			if np.abs(self.dX) > 20:
		  				self.dirX = "East" if np.sign(self.dX) == 1 else "West"

		  			# ensure there is significant movement in the
		  			# y-direction
		  			if np.abs(self.dY) > 20:
		  				self.dirY = "North" if np.sign(self.dY) == 1 else "South"

		  			# handle when both directions are non-empty
		  			if self.dirX != "" and self.dirY != "":
		  				self.direction = "{}-{}".format(self.dirY, self.dirX)

		  			# otherwise, only one direction is non-empty
		  			else:
		  				self.direction = self.dirX if self.dirX != "" else self.dirY

				# otherwise, compute the thickness of the line and
				# draw the connecting lines
				thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
				cv2.line(self.cv_image, self.pts[i - 1], self.pts[i], (0, 0, 255), 					thickness)

				self.cbInfo()
			self.counter +=1
				
			self.cbShowImage()

			# Allow up to one second to connection
			rospy.sleep(0.1)
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		try:
			rospy.logwarn("ColoredTracking (ROI) node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('color_analysis_node', anonymous=False)
	color = color_analysis_node()

	# Camera preview
	while not rospy.is_shutdown():
		color.cbFace()

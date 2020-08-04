#!/usr/bin/env python

#Title:
#Author: Nurshafikah binti Darwis & Khairul Izwan Bin Kamsani - [4-08-2020]
#Description: detect_loosefruit_node(ROS)

# import the necessary packages
import rospy
import sys
import cv2
import imutils
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
from skimage import measure
from imutils import contours


#import the necessary ROS message
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import RegionOfInterest

class detect_loosefruit_node:

    def __init__(self):
        rospy.init__node("detect_loosefruit_node")
        self.bridge = CvBridge()

        # define the lower and upper boundaries of the "loosefruit"
        #in the HSV color space, then initialize the list of tracked points
        self.lower_red = (0, 120, 70)
        self.upper_red = (10, 255, 255)
        self.lowerRed = (170, 120, 70)
        self.upperRed = (180, 255, 255)

        self.image_recieved = False

        # rospy shutdown
        rospy.on_shutdown(self.cbShutdown)

        # Subscribe to Image msg
        image_topic = "/cv_camera/image_raw"
        self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

        # Subscribe to CameraInfo msg
        cameraInfo_topic = "/cv_camera/camera_info"
        self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo,self.cbCameraInfo)

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

		cv2.imshow("Loosefruit (ROI)", self.cv_image)
		cv2.waitKey(1)

	# Image information callback
	def cbInfo(self):

		fontFruit = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

	# Detect the fruit(s)
	def cbFruit(self):
            if self.image_received:
                frame = imutils.resize(self.cv_image, width=self.imgWidth)
                gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
                mask2 = cv2.inRange(hsv, self.lowerRed, self.upperRed)
                mask = mask1 + mask2
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=1)
                for label in np.unique(labels):
                    if label == 0:
                        continue
                        labelMask = np.zeros(mask.shape, dtype="uint8")
                        labelMask[labels == label] = 255
                        numPixels = cv2.countNonZero(labelMask)
                        if numPixels > 190:
                            mask = cv2.add(mask, labelMask)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                center = None

                for (i, c) in enumerate(cnts):
                    (x, y, w, h) = cv2.boundingRect(c)
                    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                    print(x, y, w, h, image.shape[1], image.shape[0])
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi,(28,28))
                    roi = img_to_array(roi)
                    roi = roi.astype("float") / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    # (notFruit, fruit) = model.predict(roi)[0]
                    # label = "Fruit" if fruit > notFruit else "Not fruit"

                    cv2.rectangle(image, (x, y), (x + w, y + h),(255, 255, 255), 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                    self.cbInfo()
                    self.cbShowImage()

                    rospy.sleep(0.1)
            else:
                rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		try:
			rospy.logwarn(" detect_loosefruit (ROI) node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('detect_loosefruit_node', anonymous=False)
	color = detect_loosefruit_node()

	# Camera preview
	while not rospy.is_shutdown():
		color.cbFace()

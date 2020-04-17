"""main.py: Main code to train parking slots monitor"""

# Example Usage: ./cv_hw1 -i image -k clusters -m grey
# Example Usage: python cv_hw1 -i image -k clusters -m rgb


__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

def read_xml(path_xml, space_id):

       import xml.etree.ElementTree as ET

       return(image, center, theta, width, height)

def extract_roi(image, center, theta, width, height):
       '''
       Rotates OpenCV image around center with angle theta (in deg)
       then crops the image according to width and height.
       '''
       import cv2
       import numpy as np

       # Uncomment for theta in radians
       #theta *= 180/np.pi

       shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

       matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
       image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

       x = int( center[0] - width/2  )
       y = int( center[1] - height/2 )

       image = image[ y:y+height, x:x+width ]

       return image
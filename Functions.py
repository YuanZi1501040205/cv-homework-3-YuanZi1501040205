"""main.py: Main code to train parking slots monitor"""

# Example Usage: ./cv_hw1 -i image -k clusters -m grey
# Example Usage: python cv_hw1 -i image -k clusters -m rgb


__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# def analysis(path_input, numSamples):
#        import cv2
#        import xml.etree.ElementTree as ET
#        import os
#        import glob
#
#        path_data1 = str(path_input) + '/parking1a/'
#        path_data2 = str(path_input) + '/parking1b/'
#        path_data3 = str(path_input) + '/parking2/'
#        path_data = []
#        path_data.append(path_data1)
#        path_data.append(path_data2)
#        path_data.append(path_data3)
#        gt_boxes = {}
#        for root_path in path_data:
#               cloudy_folders_under_root = os.listdir(root_path + 'cloudy')
#               rainy_folders_under_root = os.listdir(root_path + 'rainy')
#               # Traverse all folders under the root folder, change the root1_folder to path_data1,2,3 respectively and rerun this cell
#               for folders in cloudy_folders_under_root:
#                      paths_image_cloudy = glob.glob(
#                             os.path.join(root_path + 'cloudy/' + folders, '*.jpg'))  # get all images in this path
#                      # Traverse all images; selected images based on the requirment number of images for testing. 25 rainy, 25 test
#                      for path in paths_image_cloudy:
#                             image = cv2.imread(path)
#                             path_xml = os.path.splitext(path)[0] + '.xml'  # read xml file associated with image
#                             tree = ET.parse(path_xml)
#                             spaces = tree.findall("space")  # read all parking slots' bounding box's informations
#                             bbx = []
#                             for space in spaces:
#                                    rotatedRect = space.find('rotatedRect')
#                                    id = space.attrib['id']
#                                    bbx.append([space.])
#                             gt_boxes.update(path:)
#               for folders in rainy_folders_under_root:
#                      rainy_image_cloudy = glob.glob(
#                             os.path.join(root_path + 'rainy/' + folders, '*.jpg'))  # get all images in this path
#        # GT Boxes
#        gt_boxes = {"img_00285.png": [[480, 457, 515, 529], [637, 435, 676, 536]]}
#        # Pred Boxes
#        pred_boxs = {"img_00285.png": {
#               "boxes": [[330, 463, 387, 505], [356, 456, 391, 521], [420, 433, 451, 498], [328, 465, 403, 540],
#                         [480, 477, 508, 522], [357, 460, 417, 537], [344, 459, 389, 493], [485, 459, 503, 511],
#                         [336, 463, 362, 496], [468, 435, 520, 521], [357, 458, 382, 485], [649, 479, 670, 531],
#                         [484, 455, 514, 519], [641, 439, 670, 532]],
#               "scores": [0.0739, 0.0843, 0.091, 0.1008, 0.1012, 0.1058, 0.1243, 0.1266, 0.1342, 0.1618, 0.2452, 0.8505,
#                          0.9113, 0.972]}}
#        import xml.etree.ElementTree as ET
#
#        return(image, center, theta, width, height)

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
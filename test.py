# coding: utf-8
"""test.py: code to test parking slots monitor"""




__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

def main():
    """ The main funtion that parses input arguments, calls the test image and models"""

    #Parse input arguments
    from argparse import ArgumentParser
    from pathlib import Path
    import sys

    parser = ArgumentParser()

    parser.add_argument("-model", type=Path, help="specify the path of the input image")
    parser.add_argument("-image", type=Path, help="Specify the the path of the output files and images")

    args = parser.parse_args()

    #Load image
    if args.path_input is None:
        print("Please specify the path of the input dataset")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        model_path = args.model


    if args.path_output is None:
        print("Please specify the path of the output dataset")
        print("use the -h option to see usage information")
    else:
        image_path = args.image


    import cv2

    car_cascade = cv2.CascadeClassifier(str(model_path))
    print('after loading model')
    img = cv2.imread(str(image_path))
    print('after loading image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("after coverting to gray")
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)
    print('after detection')
    i = 0
    for (x, y, w, h) in cars:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("bounding box",i)
        i = i + 1
    cv2.imwrite('test.jpg', img)


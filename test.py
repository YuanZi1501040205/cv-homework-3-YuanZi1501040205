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
    parser.add_argument("-scale_factor", "--scale_factor", dest="scale_factor",
                        help="Specify how much the image size is reduced at each image scale.")
    parser.add_argument("-min_neighbors", "--min_neighbors", dest="min_neighbors",
                        help="Specify Parameter specifying how many neighbors each candidate rectangle should have to retain it.")
    parser.add_argument("-min_size_w", "--min_size_w", dest="min_size_w",
                        help="Specify Minimum possible object size. Objects smaller than that are ignored")
    parser.add_argument("-min_size_h", "--min_size_h", dest="min_size_h",
                        help="Specify Minimum possible object size. Objects smaller than that are ignored")
    parser.add_argument("-maxSize_w", "--maxSize_w", dest="maxSize_w",
                        help="Specify Maximum possible object size. Objects larger than that are ignored.")
    parser.add_argument("-maxSize_h", "--maxSize_h", dest="maxSize_h",
                        help="Specify Maximum possible object size. Objects larger than that are ignored.")

    args = parser.parse_args()

    #Load image
    if args.model is None:
        print("Please specify the path of the input dataset")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        model_path = args.model


    if args.image is None:
        print("Please specify the path of the output dataset")
        print("use the -h option to see usage information")
    else:
        image_path = args.image

    if args.scale_factor is None:
        print("use the -h option to see usage information")
    else:
        scale_factor = args.scale_factor

    if args.min_neighbors is None:
        print("use the -h option to see usage information")
    else:
        min_neighbors = args.min_neighbors

    if args.min_size_w is None:
        print("use the -h option to see usage information")
    else:
        min_size_w = args.min_size_w

    if args.min_size_h is None:
        print("use the -h option to see usage information")
    else:
        min_size_h = args.min_size_h

    if args.maxSize_w is None:
        print("use the -h option to see usage information")
    else:
        maxSize_w = args.maxSize_w

    if args.maxSize_h is None:
        print("use the -h option to see usage information")
    else:
        maxSize_h = args.maxSize_h

    import cv2

    car_cascade = cv2.CascadeClassifier(str(model_path))
    print('after loading model')
    img = cv2.imread(str(image_path))
    print('after loading image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("after coverting to gray")
    cars = car_cascade.detectMultiScale(gray, scaleFactor=float(scale_factor), minNeighbors=int(min_neighbors), minSize=(int(min_size_w), int(min_size_h)), maxSize=(int(maxSize_w),int(maxSize_h)))
    print('after detection')
    i = 0
    for (x, y, w, h) in cars:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("bounding box",i)
        i = i + 1
    cv2.imwrite(str(image_path).split('/')[-1].split('.')[0] +'scale'+scale_factor+'minneighbors'+min_neighbors+'minh'+min_size_h+'minw'+min_size_w+'maxh'+maxSize_h+'maxw'+maxSize_w+ '_test.jpg', img)

if __name__ == "__main__":
    main()

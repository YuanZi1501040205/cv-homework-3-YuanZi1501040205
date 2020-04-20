"""main.py: Preprocess code to prepare train and test datasets for training"""

#Example Usage: python Extractor.py -path_input '/project/kakadiaris/yz/dataset/' -path_output '/project/kakadiaris/yz/dataset/images/' -dataset train -numPos 14000 -numNeg 10000
#Example Usage: python Extractor.py -path_input '/project/kakadiaris/yz/dataset/' -path_output '/project/kakadiaris/yz/dataset/images/' -dataset test -numPos 3500 -numNeg 2500

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

# import necessary apis
import cv2
from Functions import extract_roi
import xml.etree.ElementTree as ET
import os
import glob
import sys
from time import sleep # timing print the progress


def main():
    """ The main funtion that parses input arguments, calls the train dataset generator or test data generator and writes the output image and annotation files"""

    #Parse input arguments
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()

    parser.add_argument("-path_input", type=Path, help="specify the path of the input dataset")
    parser.add_argument("-path_output", type=Path, help="Specify the the path of the output files and images")
    parser.add_argument("-dataset", "--dataset", dest="dataset",
                        help="Specify which dataset you want to generate(train or test)", metavar="DATASET")
    parser.add_argument("-numPos", "--numPos", dest="numPos",
                        help="Specify the number of positive samples you want to generate", metavar="NUMPOS")
    parser.add_argument("-numNeg", "--numNeg", dest="numNeg",
                        help="Specify the number of negative samples you want to generate", metavar="NEGPOS")

    args = parser.parse_args()

    #Load image
    if args.path_input is None:
        print("Please specify the path of the input dataset")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        path_input = args.path_input


    if args.path_output is None:
        print("Please specify the path of the output dataset")
        print("use the -h option to see usage information")
    else:
        path_output = args.path_output

    # Check resize scale parametes
    if args.dataset is None:
        print("please specify train or test)")
        print("use the -h option to see usage information")
    elif args.dataset not in ['train', 'test']:
        print("Unknown dataset, please specify (train or test)")
        print("use the -h option to see usage information")
    elif args.dataset == 'train':
        model = 'train'
    else:
        model = 'test'

    if args.numPos is None:
        print("Please Specify the number of positive samples you want to generate")
        print("use the -h option to see usage information")
    else:
        numPos = args.numPos

    if args.numNeg is None:
        print("Please Specify the number of negative samples you want to generate")
        print("use the -h option to see usage information")
    else:
        numNeg = args.numNeg

    # configure input and output path
    path_output_train = str(path_output) + '/train/'
    path_output_test = str(path_output) + '/test/'
    path_data1 = str(path_input) + '/parking1a/'
    path_data2 = str(path_input) + '/parking1b/'
    path_data3 = str(path_input) + '/parking2/'
    path_data = []
    path_data.append(path_data1)
    path_data.append(path_data2)
    path_data.append(path_data3)
    train_annotation = [[], []]
    train_annotation_neg = []
    train_annotation_pos = []
    count = 0
    count_pos = 0
    count_neg = 0


    if model == 'train':
        # read xml and images then crop roi
        for root_path in path_data:
            folders_under_root = os.listdir(root_path + 'sunny')

            # Traverse all folders under the root folder, change the root1_folder to path_data1,2,3 respectively and rerun this cell
            for folders in folders_under_root:
                paths_image = glob.glob(
                    os.path.join(root_path + 'sunny/' + folders, '*.jpg'))  # get all images in this path

                # Traverse all images; extract rois; save them to the output folder
                for path in paths_image:
                    image = cv2.imread(path)
                    path_xml = os.path.splitext(path)[0] + '.xml'  # read xml file associated with image
                    tree = ET.parse(path_xml)
                    spaces = tree.findall("space")  # read all parking slots' bounding box's informations
                    for space in spaces:
                        rotatedRect = space.find('rotatedRect')
                        id = space.attrib['id']
                        if len(space.attrib) == 1:
                            pass
                        else:
                            occupied = space.attrib['occupied']
                        if count_neg < int(numNeg) and occupied == '0':
                            center = tuple(
                                [int(rotatedRect.getchildren()[0].attrib['x']),
                                 int(rotatedRect.getchildren()[0].attrib['y'])])
                            theta = int(rotatedRect.getchildren()[2].attrib['d'])
                            width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(
                                rotatedRect.getchildren()[1].attrib['h'])

                            # cropping roi based on parking slot space information
                            roi = extract_roi(image, center, theta, width, height)
                            if roi.shape[0] != 0 and roi.shape[1] != 0 and roi.shape[2] != 0:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                roi_gray = cv2.resize(roi_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

                                # build the annotation file with xml record
                                train_annotation[0].append(
                                    root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                        0] + '_no' + id + '_slots.jpg')
                                train_annotation[1].append(int(occupied))

                                # save negative sample respectively and create .txt file for both.

                                # save gray scale resized roi images to pos folder
                                cv2.imwrite(path_output_train + 'neg/' + root_path.split("/")[-2] + '_' +
                                            path.split("/")[-1].split("\\")[-1].split(".")[
                                                0] + '_no' + id + '_slots.jpg', roi_gray)
                                train_annotation_neg.append(
                                    root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                        0] + '_no' + id + '_slots.jpg')
                                count_neg = count_neg + 1
                                count = count + 1
                                print('number of negatives: ', count_neg)
                            else:
                                pass
                        elif count_pos < int(numPos) and occupied == '1':
                            rotatedRect = space.find('rotatedRect')
                            id = space.attrib['id']
                            if len(space.attrib) == 1:
                                pass
                            else:
                                occupied = space.attrib['occupied']

                                center = tuple(
                                    [int(rotatedRect.getchildren()[0].attrib['x']),
                                     int(rotatedRect.getchildren()[0].attrib['y'])])
                                theta = int(rotatedRect.getchildren()[2].attrib['d'])
                                width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(
                                    rotatedRect.getchildren()[1].attrib['h'])

                                # cropping roi based on parking slot space information
                                roi = extract_roi(image, center, theta, width, height)
                                if roi.shape[0] != 0 and roi.shape[1] != 0 and roi.shape[2] != 0:

                                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    roi_gray = cv2.resize(roi_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

                                    # build the annotation file with xml record
                                    train_annotation[0].append(
                                        root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                            0] + '_no' + id + '_slots.jpg')
                                    train_annotation[1].append(int(occupied))

                                    # save positive samples and negative sample respectively and create .txt file for both.
                                    # save gray scale resized roi images to pos folder
                                    cv2.imwrite(path_output_train + 'pos/' + root_path.split("/")[-2] + '_' +
                                                path.split("/")[-1].split("\\")[-1].split(".")[
                                                    0] + '_no' + id + '_slots.jpg', roi_gray)
                                    train_annotation_pos.append([root_path.split("/")[-2] + '_' +
                                                                 path.split("/")[-1].split("\\")[-1].split(".")[
                                                                     0] + '_no' + id + '_slots.jpg ', '1 ', '0 ', '0 ',
                                                                 '24 ',
                                                                 '24'])
                                    count_pos = count_pos + 1
                                    count = count + 1
                                    print('number of negatives: ', count_pos)
                                else:
                                    pass
                        else:
                            pass

        # Write txt file for positive and negative samples' annotation
        f = open(path_output_train + 'pos/positive.txt', 'w')
        for i in range(len(train_annotation_pos)):
            for j in range(6):
                f.write(train_annotation_pos[i][j])
            f.write('\n')
        f.close()
        # f = open('/project/kakadiaris/yz/dataset/images/train/negative.txt', 'w')
        # for i in range(len(train_annotation_neg)):
        #     f.write(train_annotation_neg[i])
        #     f.write('\n')
        # f.close()
        print('negative', count_neg)
        print('positive', count_pos)
        print('Done!!!')

    elif model == 'test':
        # read xml and images then crop roi
        for root_path in path_data:
            folders_under_root = os.listdir(root_path + 'cloudy')

            # Traverse all folders under the root folder, change the root1_folder to path_data1,2,3 respectively and rerun this cell
            for folders in folders_under_root:
                paths_image = glob.glob(
                    os.path.join(root_path + 'cloudy/' + folders, '*.jpg'))  # get all images in this path

                # Traverse all images; extract rois; save them to the output folder
                for path in paths_image:
                    image = cv2.imread(path)
                    path_xml = os.path.splitext(path)[0] + '.xml'  # read xml file associated with image
                    tree = ET.parse(path_xml)
                    spaces = tree.findall("space")  # read all parking slots' bounding box's informations
                    for space in spaces:
                        rotatedRect = space.find('rotatedRect')
                        id = space.attrib['id']
                        if len(space.attrib) == 1:
                            pass
                        else:
                            occupied = space.attrib['occupied']
                        if count_neg < int(numNeg) and occupied == '0':
                            center = tuple(
                                [int(rotatedRect.getchildren()[0].attrib['x']),
                                 int(rotatedRect.getchildren()[0].attrib['y'])])
                            theta = int(rotatedRect.getchildren()[2].attrib['d'])
                            width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(
                                rotatedRect.getchildren()[1].attrib['h'])

                            # cropping roi based on parking slot space information
                            roi = extract_roi(image, center, theta, width, height)
                            if roi.shape[0] != 0 and roi.shape[1] != 0 and roi.shape[2] != 0:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                roi_gray = cv2.resize(roi_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

                                # build the annotation file with xml record
                                train_annotation[0].append(
                                    root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                        0] + '_no' + id + '_slots.jpg')
                                train_annotation[1].append(int(occupied))

                                # save negative sample respectively and create .txt file for both.

                                # save gray scale resized roi images to pos folder
                                cv2.imwrite(path_output_test + 'neg/' + root_path.split("/")[-2] + '_' +
                                            path.split("/")[-1].split("\\")[-1].split(".")[
                                                0] + '_no' + id + '_slots.jpg', roi_gray)
                                train_annotation_neg.append(
                                    root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                        0] + '_no' + id + '_slots.jpg')
                                count_neg = count_neg + 1
                                count = count + 1
                                print('number of negatives: ', count_neg)
                            else:
                                pass
                        elif count_pos < int(numPos) and occupied == '1':
                            rotatedRect = space.find('rotatedRect')
                            id = space.attrib['id']
                            if len(space.attrib) == 1:
                                pass
                            else:
                                occupied = space.attrib['occupied']

                                center = tuple(
                                    [int(rotatedRect.getchildren()[0].attrib['x']),
                                     int(rotatedRect.getchildren()[0].attrib['y'])])
                                theta = int(rotatedRect.getchildren()[2].attrib['d'])
                                width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(
                                    rotatedRect.getchildren()[1].attrib['h'])

                                # cropping roi based on parking slot space information
                                roi = extract_roi(image, center, theta, width, height)
                                if roi.shape[0] != 0 and roi.shape[1] != 0 and roi.shape[2] != 0:

                                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    roi_gray = cv2.resize(roi_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

                                    # build the annotation file with xml record
                                    train_annotation[0].append(
                                        root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[
                                            0] + '_no' + id + '_slots.jpg')
                                    train_annotation[1].append(int(occupied))

                                    # save positive samples and negative sample respectively and create .txt file for both.
                                    # save gray scale resized roi images to pos folder
                                    cv2.imwrite(path_output_test + 'pos/'+ root_path.split("/")[-2] + '_' +
                                                path.split("/")[-1].split("\\")[-1].split(".")[
                                                    0] + '_no' + id + '_slots.jpg', roi_gray)
                                    train_annotation_pos.append([root_path.split("/")[-2] + '_' +
                                                                 path.split("/")[-1].split("\\")[-1].split(".")[
                                                                     0] + '_no' + id + '_slots.jpg ', '1 ', '0 ', '0 ',
                                                                 '24 ',
                                                                 '24'])
                                    count_pos = count_pos + 1
                                    count = count + 1
                                    print('number of negatives: ', count_pos)
                                else:
                                    pass
                        else:
                            pass

        # Write txt file for positive and negative samples' annotation
        f = open(path_output_test + 'pos/positive.txt', 'w')
        for i in range(len(train_annotation_pos)):
            for j in range(6):
                f.write(train_annotation_pos[i][j])
            f.write('\n')
        f.close()
        # f = open('/project/kakadiaris/yz/dataset/images/train/negative.txt', 'w')
        # for i in range(len(train_annotation_neg)):
        #     f.write(train_annotation_neg[i])
        #     f.write('\n')
        # f.close()
        print('negative', count_neg)
        print('positive', count_pos)
        print('Done!!!')
    else:
        print('fail to run!')
        pass


if __name__ == "__main__":
    main()


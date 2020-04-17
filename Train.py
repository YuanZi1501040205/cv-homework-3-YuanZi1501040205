# %% import necessary apis
import cv2
from Functions import extract_roi
import xml.etree.ElementTree as ET
import os
import glob
from time import sleep # timing print the progress


# configure input and output path
path_output = 'D:/DATA/CVHW3_PARKING_LOTS_Surveillance/output/'
path_output_train = path_output + 'train/'
path_output_test = path_output + 'test/'
path_data1 = 'D:/DATA/CVHW3_PARKING_LOTS_Surveillance/parking1a/'
path_data2 = 'D:/DATA/CVHW3_PARKING_LOTS_Surveillance/parking1b/'
path_data3 = 'D:/DATA/CVHW3_PARKING_LOTS_Surveillance/parking2/'
path_data = []
path_data.append(path_data1)
path_data.append(path_data2)
path_data.append(path_data3)
train_annotation = [[],[]]
train_annotation_neg = []
train_annotation_pos = []
count = 0
count_pos = 0
count_neg = 0


# read xml and images then crop roi
for root_path in path_data:
    folders_under_root = os.listdir(root_path+'sunny')

    # Traverse all folders under the root folder, change the root1_folder to path_data1,2,3 respectively and rerun this cell
    for folders in folders_under_root:
        paths_image = glob.glob(os.path.join(root_path + 'sunny/' + folders , '*.jpg'))# get all images in this path

        # Traverse all images; extract rois; save them to the output folder
        for path in paths_image:
            image = cv2.imread(path)
            path_xml = os.path.splitext(path)[0]+'.xml'# read xml file associated with image
            tree = ET.parse(path_xml)
            spaces = tree.findall("space")# read all parking slots' bounding box's informations
            for space in spaces:
                rotatedRect = space.find('rotatedRect')
                contour = space.find('contour')
                id = space.attrib['id']
                if len(space.attrib) == 1:
                    pass
                else:
                    occupied = space.attrib['occupied']
                    if count_pos<14000 or count_neg<10000:
                        center = tuple(
                            [int(rotatedRect.getchildren()[0].attrib['x']), int(rotatedRect.getchildren()[0].attrib['y'])])
                        theta = int(rotatedRect.getchildren()[2].attrib['d'])
                        width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(rotatedRect.getchildren()[1].attrib['h'])

                        # cropping roi based on parking slot space information
                        roi = extract_roi(image, center, theta, width, height)
                        if roi.shape[0] !=0 and roi.shape[1] != 0 and roi.shape[2] !=0:
                            roi = cv2.resize(roi, (45, 50), interpolation=cv2.INTER_CUBIC)
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                            # build the annotation file with xml record
                            train_annotation[0].append(root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[0] + '_no' + id +'_slots.jpg')
                            train_annotation[1].append(int(occupied))

                            # save positive samples and negative sample respectively and create .txt file for both.
                            if occupied == '1':
                                # save gray scale resized roi images to pos folder
                                cv2.imwrite('./dataset/images/train/pos/' + root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[0] + '_no' + id + '_slots.jpg', roi)
                                train_annotation_pos.append([root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[0] + '_no' + id + '_slots.jpg', '1', '0', '0', '45', '50'])
                                count_pos = count_pos + 1
                            else:
                                # save gray scale resized roi images to pos folder
                                cv2.imwrite('./dataset/images/train/neg/' + root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[0] + '_no' + id + '_slots.jpg', roi)
                                train_annotation_neg.append([root_path.split("/")[-2] + '_' + path.split("/")[-1].split("\\")[-1].split(".")[0] + '_no' + id + '_slots.jpg'])
                                count_neg = count_neg + 1
                            count = count + 1
                            sleep(0.02)
                            print('\r', 'count：{:.2f}%'.format(count, end='', flush=True))
                        else:
                            pass
                    else:
                        pass
# Write txt file for positive and negative samples' annotation
f = open('/project/kakadiaris/yz/dataset/images/train/pos/positive.txt','w')
for i in range(len(train_annotation_pos)):
    for j in range(6):
        f.write(train_annotation_pos[i][j]+' ')
    f.write('\n')
f.close()
f = open('/project/kakadiaris/yz/dataset/images/train/neg/negative.txt', 'w')
for i in range(len(train_annotation_neg)):
    f.write(train_annotation_pos[i] + '\n')
f.close()
print('negative', count_neg)
print('positive', count_pos)
print('Done!!!')

# %%
tree = ET.parse(path_data + '/2012-09-11_15_16_58.xml')
image = cv2.imread(path_data + '/2012-09-11_15_16_58.jpg')
spaces = tree.findall("space")
# # print details of spaces
n = 1
sum = 0
for space in spaces:
    rotatedRect = space.find('rotatedRect')
    contour = space.find('contour')
    id = space.attrib['id']
    # occupied = space.attrib['occupied']
    x1 = contour.getchildren()[0].attrib['x']
    center = tuple([int(rotatedRect.getchildren()[0].attrib['x']), int(rotatedRect.getchildren()[0].attrib['y'])])
    theta = -int(rotatedRect.getchildren()[2].attrib['d'])/np.pi
    width, height = int(rotatedRect.getchildren()[1].attrib['w']), int(rotatedRect.getchildren()[1].attrib['h'])
    # cropping roi based on parking slot space information
    roi = extract_roi(image, center, theta, width, height)
    roi = cv2.resize(roi, (44, 51), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(id+'_sample.jpg', roi)

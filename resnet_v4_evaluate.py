import os
import cv2
import numpy as np
from keras import layers
import csv
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
# from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
# # from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# import keras.backend as K
# import tensorflow as tf
train_acl_lbl = []
train_abnormal_lbl = []
train_meniscus_lbl = []

valid_acl_lbl = []
valid_abnormal_lbl = []
valid_meniscus_lbl = []

i = 0
csvs = ["train-acl.csv", "train-abnormal.csv", "train-meniscus.csv", "valid-acl.csv", "valid-abnormal.csv",
        "valid-meniscus.csv"]
for c in csvs:
    with open('/home/vvarier/ai_project/MRNet-v1.0/' + c, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar='|')
        if i == 0:
            for row in read:
                train_acl_lbl.append(', '.join(row)[5])
                # print "reaching here"
            i += 1
        elif i == 1:
            for row in read:
                train_abnormal_lbl.append(', '.join(row)[5])
            i += 1
        elif i == 2:
            for row in read:
                train_meniscus_lbl.append(', '.join(row)[5])
            i += 1
        elif i == 3:
            for row in read:
                valid_acl_lbl.append(', '.join(row)[5])
            i += 1
        elif i == 4:
            for row in read:
                valid_abnormal_lbl.append(', '.join(row)[5])
            i += 1
        elif i == 5:
            for row in read:
                valid_meniscus_lbl.append(', '.join(row)[5])
            i += 1


def getTheLabels(a, b, c):
    # print "length of each is ", len(a), len(b), len(c)
    labels = [0] * len(a)
    for i in range(len(labels)):
        # print "val is ", str(a[i]) + str(c[i])
        labels[i] = int(str(a[i]) + str(b[i]) + str(c[i]), 2)
    return np.array(labels)


# print "train acl lbl ", train_acl_lbl
# Encoding all labels to be a number from (0-7) (Abnormal,ACL,Meniscus)
# GAN doesn't look like using labels
train_label = getTheLabels(train_abnormal_lbl, train_acl_lbl, train_meniscus_lbl)
valid_label = getTheLabels(valid_abnormal_lbl, valid_acl_lbl, valid_meniscus_lbl)
del (train_abnormal_lbl)
del (train_acl_lbl)
del (train_meniscus_lbl)
del (valid_abnormal_lbl)
del (valid_acl_lbl)
del (valid_meniscus_lbl)

WIDTH = 256
HEIGHT = 256

# load x_train
train_axial = np.zeros([38778, WIDTH, HEIGHT], dtype='uint8')
train_coronal = np.zeros([33649, WIDTH, HEIGHT], dtype='uint8')
train_sagittal = np.zeros([34370, WIDTH, HEIGHT], dtype='uint8')

train_axial_lbl = np.zeros([38778], dtype='uint8')
train_coronal_lbl = np.zeros([33649], dtype='uint8')
train_sagittal_lbl = np.zeros([34370], dtype='uint8')

train_axial_idx = 0
train_sagittal_idx = 0
train_coronal_idx = 0

dir_train = "/home/vvarier/ai_project/MRNet-v1.0/train"


def to_rgb(img, wid, hei):  # -> Resizing image to fit as (WIDTH,HEIGHT)
    img = cv2.resize(img, (wid, hei), interpolation=cv2.INTER_AREA)
    return img


def getTheDataLabelPerView_(obj, save_in, idx):
    global train_axial_idx, train_sagittal_idx, train_coronal_idx
    for j in range(len(obj)):  # 0 -> s (For every view)
        if (save_in == 'train_axial'):
            train_axial[train_axial_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
            train_axial_lbl[train_axial_idx] = train_label[idx]  # -> Giving all images the same label as patient.
            train_axial_idx += 1
        elif (save_in == 'train_coronal'):
            train_coronal[train_coronal_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
            train_coronal_lbl[train_coronal_idx] = train_label[idx]  # -> Giving all images the same label as patient.
            train_coronal_idx += 1
        else:
            train_sagittal[train_sagittal_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
            train_sagittal_lbl[train_sagittal_idx] = train_label[idx]  # -> Giving all images the same label as patient.
            train_sagittal_idx += 1


i = 0
for folder in sorted(os.listdir(dir_train)):
    idx = 0
    if folder == ".DS_Store" or folder == 'DG1__DS_DIR_HDR':
        continue
    type_dir = os.path.join(dir_train, folder)
    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store" or img == 'DG1__DS_DIR_HDR':
            continue
        img_dir = os.path.join(type_dir, img)
        if i == 0:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_axial', idx)
        elif i == 1:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_coronal', idx)
        elif i == 2:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_sagittal', idx)

        idx += 1
    i += 1

# load y_train
valid_ = []

dir_valid = "/home/vvarier/ai_project/MRNet-v1.0/train"
i = 0
for folder in sorted(os.listdir(dir_valid)):
    if folder == ".DS_Store":
        continue
    type_dir = os.path.join(dir_valid, folder)

    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store":
            continue
        img_dir = os.path.join(type_dir, img)

        if i == 0:
            valid_.append(np.load(img_dir).astype('uint8'))
        elif i == 1:
            valid_.append(np.load(img_dir).astype('uint8'))

    i += 1



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

train_axial_lbl = train_axial_lbl.reshape(38778, 1)
# print("label shape ", train_axial_lbl.shape)

x_train, x_test, y_train, y_test = train_test_split(train_axial, train_axial_lbl, test_size=0.2, random_state=42)

x_train = np.stack([x_train], axis=-1)
x_test = np.stack([x_test], axis=-1)

ROWS, COLS, CHANNELS = x_train.shape[1:]
CLASSES = 8

train_set_x, train_set_y = x_train, y_train
test_set_x, test_set_y = x_test, y_test

X_train = train_set_x/255
X_test = test_set_x/255

Y_train = convert_to_one_hot(train_set_y, CLASSES).T
Y_test = convert_to_one_hot(test_set_y, CLASSES).T

axial = np.load("/home/vvarier/ai_project/MRNet-v1.0/valid/axial/1181.npy")
coronal = np.load("/home/vvarier/ai_project/MRNet-v1.0/valid/coronal/1181.npy")
sagittal = np.load("/home/vvarier/ai_project/MRNet-v1.0/valid/sagittal/1181.npy")


model_a = load_model('/home/vvarier/ai_project/output_file/weights_file/ResNet50.h5')
model_a.load_weights('/home/vvarier/ai_project/output_file/weights-improvement-50-0.76.hdf5')
model_s = load_model('/home/vvarier/ai_project/output_file/weights_file/ResNet50_sagittal.h5')
model_s.load_weights('/home/vvarier/ai_project/output_file/sagittal/weights-improvement-35-0.54.hdf5')
model_c = load_model('/home/vvarier/ai_project/output_file/weights_file/ResNet50_coronal.h5')
model_c.load_weights('/home/vvarier/ai_project/output_file/coronal/weights-improvement-21-0.43.hdf5')

print("ResNet50 Model Predicted Results")
print("The accuracy of axial plane model is 76%, sagittal plane model is 54% and coronal plane model is 43%")
print("For the user 1181, the predicted label of each of the image plane is:")
pred_image_a = model.predict(axial[0, :, :].reshape((1, 256,256,1)), batch_size=1)
cc_a = np.argmax(pred_image_a)
print("The predicted class label for axial plane image is", cc_a)
pred_image_s = model.predict(sagittal[0, :, :].reshape((1, 256,256,1)), batch_size=1)
cc_s = np.argmax(pred_image_s)
print("The predicted class label for sagittal plane image is", cc_s)
pred_image_c = model.predict(coronal[0, :, :].reshape((1, 256,256,1)), batch_size=1)
cc_c = np.argmax(pred_image_c)
print("The predicted class label for coronal plane image is", cc_c)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
ax1.imshow(axial[0, :, :], 'gray')
ax1.set_title('Case 1181|Axial|Predicted Class: {}'.format(cc_a))

ax2.imshow(coronal[0, :, :], 'gray')
ax2.set_title('Case 1181|Coronal Predicted Class: {}'.format(cc_c))

ax3.imshow(sagittal[0, :, :], 'gray')
ax3.set_title('Case 1181|Sagittal|Predicted Class: {}'.format(cc_s))
fig.savefig('/home/vvarier/ai_project/results_resnet.png')


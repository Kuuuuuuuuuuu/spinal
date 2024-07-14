import os, glob
import numpy as np
from PIL import Image
from numpy import argmax

image_w=380
image_h=380

data_path = 'drive/My Drive/Test1'
train_data_path = os.path.join(data_path, 'Train_pro')
test_data_path = os.path.join(data_path, 'Test_pro')
images = os.listdir(train_data_path)
total = len(images)
categories = ["Normal","SC"]
nb_classes = len(categories)
print("1")
X = []
Y = []

X1 = []
Y1 = []

def create_train_data():
    for idx, cat in enumerate(categories):
        label = [0 for i in range(nb_classes)]
        label[idx] = 1

        image_dir = train_data_path + "/" + cat
        files = glob.glob(image_dir+"/*.bmp") # 파일 이름 다불러옴 확장자가 bmp인 모든 파일

        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w,image_h))
            data = np.asarray(img)

            X.append(data)
            Y.append(label)

    np.save('imgs_train.npy', X)
    np.save('label_train.npy', Y)

def create_test_data():
    for idx, cat in enumerate(categories):
        label = [0 for i in range(nb_classes)]
        label[idx] = 1

        image_dir = test_data_path + "/" + cat
        files = glob.glob(image_dir+"/*.bmp") # 파일 이름 다불러옴

        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w,image_h))
            data = np.asarray(img)

            X1.append(data)
            Y1.append(label)

    np.save('imgs_test.npy', X1)
    np.save('label_test.npy', Y1)

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    label_train = np.load('label_train.npy')
    return imgs_train, label_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    label_test = np.load('label_test.npy')
    return imgs_test, label_test

create_test_data()
create_train_data()
(X_train, Y_train) = load_train_data()
(X_test, Y_test) = load_test_data()
print(X_train.shape, X_train.dtype)
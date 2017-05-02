import numpy as np
import os
from PIL import Image
import sys

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess(img):
    avg = np.mean(img)
    dev = np.std(img)

    img -= avg
    img /= dev

    return img

def load_cifar():
    # Constants
    TRAIN_FILES = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    TEST_FILE = "test_batch"
    DIRNAME = "cifar/"
    BATCH_SIZE = 10000
    TRAIN_NO = len(TRAIN_FILES) * BATCH_SIZE

    data = {}
    train_img = np.empty((TRAIN_NO, 3, 32, 32), dtype='float64')
    train_labels = np.empty(TRAIN_NO)

    # Open and add each batch to the array
    for idx, f in enumerate(TRAIN_FILES):
        img = unpickle(DIRNAME + f)
        labels = img[b'labels']
        img = img[b'data'].reshape(BATCH_SIZE, 3, 32, 32)
        img = img.astype('float')

        for i in range(BATCH_SIZE):
            img[i] = preprocess(img[i])

        train_img[(idx * BATCH_SIZE) : (idx + 1) * BATCH_SIZE] = img
        train_labels[(idx * BATCH_SIZE) : (idx + 1) * BATCH_SIZE] = labels
            
    data["train_imgs"] = train_img
    data["train_labels"] = train_labels

    img = unpickle(DIRNAME + f)
    labels = img[b'labels']
    img = img[b'data'].reshape(BATCH_SIZE, 3, 32, 32)
    img = img.astype('float')

    # Standardize images
    for i in range(BATCH_SIZE):
        img[i] = preprocess(img[i])
    
    data["test_imgs"] = img
    data["test_labels"] = labels
    data["train_no"] = TRAIN_NO
    data["test_no"] = BATCH_SIZE

    return data

def load_julia():
    # Constants
    DIRNAME = 'julia/'
    SUBDIRS = ['GoodImg/Bmp', 'BadImg/Bmp']
    TRAIN_IMG_NO = 700
    TEST_IMG_NO = 185

    data = {}
    train_img = []
    test_img = []
    train_labels = []
    test_labels = []
    img_idx = 0

    # Open each subdir
    for subdir in SUBDIRS:
        folders_path = os.path.join(DIRNAME, subdir)

        folders = [f for f in os.listdir(folders_path)][:10]

        for class_index, class_folder in enumerate(folders):
            class_path = os.path.join(folders_path, class_folder)
            files = [f for f in os.listdir(class_path)]

            for image_file in files:
                image = Image.open(os.path.join(class_path, image_file))
                image_array = np.array(image.resize((32, 32)), dtype='float64')

                if img_idx < TRAIN_IMG_NO:
                    train_img.append(image_array)
                    train_labels.append(class_index)
                else:
                    test_img.append(image_array)
                    test_labels.append(class_index)

                img_idx += 1

    train_img = np.array(train_img).swapaxes(1, 3)
    test_img = np.array(test_img).swapaxes(1, 3)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    for i in range(TRAIN_IMG_NO):
        train_img[i] = preprocess(train_img[i])
    for i in range(TEST_IMG_NO):
        test_img[i] = preprocess(test_img[i])

    data["train_imgs"] = train_img
    data["train_labels"] = train_labels
    data["test_imgs"] = test_img
    data["test_labels"] = test_labels
    data["train_no"] = TRAIN_IMG_NO
    data["test_no"] = TEST_IMG_NO

    return data

if __name__ == "__main__":
    print(load_julia()['train_imgs'])

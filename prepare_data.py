import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

TRAIN_FILES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]
TEST_FILE = "test_batch"
DIRNAME = "cifar-10/"
BATCH_SIZE = 10000
TRAIN_NO = len(TRAIN_FILES) * BATCH_SIZE

def preprocess(img):
    avg = np.mean(img)
    dev = np.std(img)

    img -= avg
    img /= dev

    return img

def load_data():
    data = {}
    train_img = np.empty((TRAIN_NO, 3, 32, 32), dtype='float64')
    train_labels = np.empty(TRAIN_NO)

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

    for i in range(BATCH_SIZE):
        img[i] = preprocess(img[i])
    
    data["test_imgs"] = img
    data["test_labels"] = labels
    data["train_no"] = TRAIN_NO
    data["test_no"] = BATCH_SIZE

    return data

if __name__ == "__main__":
    load_data()

import numpy as np
import os
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from utils import to_one_hot
def load_data(data_dir):
    X = []
    y = []
    
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for label, folder in enumerate(subfolders):
        for file_name in os.listdir(folder):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image = Image.open(os.path.join(folder, file_name))
                image_array = np.array(image)
                X.append(image_array)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    classes = os.listdir(data_dir)
    np.random.seed(0)
    np.random.shuffle(X)
    np.random.seed(0)
    np.random.shuffle(y)
    return X, y, classes
# data_dir_train = 'D:/Workspace/pythonProject1/DL_University/character/train'
# data_dir_test = 'D:/Workspace/pythonProject1/DL_University/character/val'
# X_train,y_train,classes=load_data(data_dir_train)
# X_test,y_test,_=load_data(data_dir_test)
# y_train=y_train.reshape(y_train.shape[0],1)
# y_test=y_test.reshape(y_test.shape[0],1)
# print(y_train)
# print(y_train.shape,X_train.shape,y_test.shape)
# # y_train = to_one_hot(y_train, len(classes))
# print(y_train)
# print(np.argmax(y_train,axis=1))
# X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
# X_test=X_test.reshape(X_test.shape[0],-1)/255

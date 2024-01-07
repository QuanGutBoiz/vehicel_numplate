import cv2
from model import Model

model = Model()
model.load('D:/Workspace/pythonProject1/Midterm_TranAnhQuan/save_model.npy')

img = cv2.imread('D:/Workspace/pythonProject1/Midterm_TranAnhQuan/character/val/class_D/class_D_12.jpg')
dic={}
character="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i,c in enumerate(character):
    dic[i]=c

img = img.reshape(-1, 28*28*3) / 255.0

print('Predicted Class: {}'.format(dic[model.predict(img)[0]]))
import numpy as np
from model import Model
from character import load_data
from utils import to_one_hot
from matplotlib import pyplot as plt

data_dir_train = 'D:/Workspace/pythonProject1/Midterm_TranAnhQuan/character/train'
data_dir_test = 'D:/Workspace/pythonProject1/Midterm_TranAnhQuan/character/val'
X_train,y_train,classes=load_data(data_dir_train)
X_test,y_test,_=load_data(data_dir_test)

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

y_train = to_one_hot(y_train, len(classes))
y_test = to_one_hot(y_test, len(classes))

model = Model()
model.fit(X_train, y_train, X_test, y_test, epochs = 1000, lr = 0.3, verbose=True)
model.save('save_model')
# model.show(X_train,y_train,X_test,y_test,True)
fig, ax = plt.subplots(1, 2)



# ax[0].plot(model.loss_history)
# ax[0].set_title('Loss History')
# ax[0].set_xlabel('Epoch')
# ax[0].set_ylabel('Loss')

# ax[1].plot(model.train_acc_history, label='Train')
# ax[1].plot(model.test_acc_history, label='Test')
# ax[1].set_title('Accuracy History')
# ax[1].set_xlabel('Epoch')
# ax[1].set_ylabel('Accuracy')
# plt.savefig('character.png')
# plt.legend()
# plt.show()
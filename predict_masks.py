import random

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from skimage.io import imshow
from skimage.transform import resize

from const import *
from utils import dice_coef, dice_coef_loss, image_processing

X_train, Y_train, X_test = image_processing()

# Предсказываем, основываясь на обучающей
model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Отсеиваем прогнозы
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Создайте список тестовых масок с повышенной дискретизацией
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

# for i in range(10):
#     # Выполните проверку работоспособности на некоторых случайных обучающих выборках
#     ix = random.randint(0, len(preds_train_t))
#     imshow(X_train[ix])
#     plt.show()
#     imshow(np.squeeze(Y_train[ix]))
#     plt.show()
#     imshow(np.squeeze(preds_train_t[ix]))
#     plt.show()


for i in range(10):
    # Выполним проверку работоспособности на некоторых случайных тестовых выборках
    ix = random.randint(0, len(preds_test_t))
    imshow(X_test[ix])
    plt.show()
    imshow(np.squeeze(preds_test_t[ix]))
    plt.show()

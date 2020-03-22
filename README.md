# UNet-Quantum-Task
Используемая архитектура нейронной сети - это так называемая U-Net, которая используется для сегментации изображений ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)).

![U-Net](https://github.com/dexXxed/UNet-Quantum-Task/blob/master/img/u-net-architecture.png?raw=true)

Датасет взят [тут](https://www.kaggle.com/c/data-science-bowl-2018/data)

# Цель данного таска

Build a **semantic** segmentation model with UNet architecture using Keras.

# Результаты, полученные после обучения модели
Данные из тестовой выборки (изначальное изображение и предсказанная сегментация)

![Исходное изображение](https://i.imgur.com/2W5wWse.png)
![Сегментированное](https://i.imgur.com/GBCOUXO.png)
![Исходное изображение](https://i.imgur.com/kuZuWEB.png)
![Сегментированное](https://i.imgur.com/oFk2xrF.png)

# Обучение модели
Использовалась метрика меры Сёренсена в Keras-е. [Тут больше информации](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
```
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
```

# Что за файлы в репозитории?
Файлов, которые содержаться в репозитории:

```input/``` - в данной директории находится обучающая и тестовая выборки

```const.py``` - вынесенны костанты данного проекта

```main_logic.ipynb``` -  IPython Notebook с основной логикой работы всего, описанного в ```.py``` файлах

```model.h5``` - обученная модель

```predict_masks.py``` - скрипт, который выдает значения обводок по тестовой выборке

```requiremets.txt``` - текстовый файл зависимостей

```train.py``` - скрипт для обучения НС

```utils.py``` - скрипт, в который вынесены функции для удобного вызова


# Как это запустить?
Первоначально выполните команду:
```
git clone https://github.com/dexXxed/UNet-Quantum-Task.git
cd UNet-Quantum-Task
```
Далее желательно использование ```Python 3.7.*``` и  модуля ```virtualenv```, установленного в системном интерпретаторе
```
virtualenv venv
source venv/bin/activate
```
После этого поставим зависимости проекта:
```
pip install -r requirements.txt
```

Для запуска **обучения** модели:
```
python train.py
```

**Для просмотра результатов** для уже обученной модели:
```
python predict_masks.py
```

# License

Copyright (c) Denys Shcherbyna. All rights reserved.

Licensed under the [MIT](LICENSE) License.

import numpy as  np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

# делим наш датасет на обучающую и тестовую выборку
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

# добавляем классы данных
class_names = [ 'T-shirt/top' , 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle bot' ]

#делаем предварительную обработку данных


# посмотрим как выглядят наши изображения

# plt.figure()
# plt.imshow(X_train[0])
# plt.colorbar()
# plr.grid(False)

# создаем модели нейронной сети

model = keras.Sequential([
                            keras.layers.Flatten(input_shape=(28,28)),   # этот слой преобразует изображение двухмерного массива в одномерный массив
                            keras.layers.Dense(128, activation="relu"),   # нейрон входной полосвязный - 128 нейронов самое удачное количество , в каждый нейрон будет поступать изображение для оценки
                            keras.layers.Dense(10, activation="softmax") # этот слой выходной - здесь количество нейронов равно колличеству наших классов одежды (class_names) 
                              #- будет возвращать массив, в котором будет выставлены оценки, в сумме которые будут равны одному
                              # в котором вес будет разбит по категориям
])

# компиляция модели
model.compile(optimize==tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # подключаем потимизатор SGD() можно использовать ADAM(), 
 # loss категориальная перекрестная энтропия - эта функция ошибки хорошо работает с задачами классификации когда классов больше двух
 # параметр metrics - качество - доля правильных ответов

model.summary()

# обучение модели

model.fit(x_train, y_train, epochs=10) # так как у нас обучение с учителем - мы передаем функции как обучающую выборку x_train, 
 # так и ответы y_train и задаем количество эпох - одна эпоха это когда весь набор данных, проходит через нашу нейронную сеть 1 раз
 # чем больше разношерстный наш набор данных - тем может потребоваться большее количество эпох


 # проверка точности предсказаний 

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accurasy' , test_acc)

# предсказывания что на рисунке

prediction = model.predict(x_train)

print(prediction[0]) # число - это номер изображения в коллекции наших изображений

nfnfnf = np.argmax(prediction[0]) # тут модель прдсказывает к какому классу изображение относится

print(nfnfnf)

print(y_train[0]) # правильный ответ

print(class_names[np.argmax(prediction[12])]) # будет выводить название класса к какому изображение относится
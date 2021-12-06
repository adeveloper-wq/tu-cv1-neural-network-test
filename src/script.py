'''Computer Vision, TUD 2021
Perceptron on data3.csv with keras.
'''
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#from keras.layers import Conv2D, MaxPooling2D 
batch_size = 100
num_classes = 2
epochs = 1000
# load the data: data3.csv, split between train and test sets
csvfile = 'Data3.csv'
print('Reading "' + csvfile + '":')
dat = np.loadtxt(csvfile, delimiter=';')
x_train = dat[:,:2]
y_train = dat[:,2:]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
model = Sequential()
model.add(Dense(2))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
print("initial w: ",model.get_weights())
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
print(model.get_weights())
p = model.get_weights()
print("w: ",p)
w=[0,0,0]
w[0]=p[0][0]
w[1]=p[0][1]
w[2]=p[1]
for i, d in enumerate(dat):
    if d[2] > 0.0:
        plt.plot(d[0], d[1], 'bo')
    else:
        plt.plot(d[0], d[1],color='#FF8000', marker='o')
# show w-line
seplin_x = [0.0, -w[2]/w[0]]
seplin_y = [-w[2]/w[1], 0.0]
plt.plot(seplin_x, seplin_y, color='red', alpha=0.4)
 
plt.title('perceptron result with keras (red) and GT (grey)')
plt.ylabel('Class 1 = Blue, Class -1 = Orange')
plt.show()
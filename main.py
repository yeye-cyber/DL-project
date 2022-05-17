
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from math import *


#för att kunna ladda ner datasetet om det inte funkar testa: pip install ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

Ini = 'he_uniform'
#Ini = tf.keras.initializers.RandomNormal(mean = 0.0, stddev=1.0)

def see_data():
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

    for i in range(9):
        plt.subplot(330+1+i)
        plt.imshow(trainX[i])
    plt.show()

def load_data():
    #fetch data
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    #onehot
    trainY = tf.keras.utils.to_categorical(trainy)
    testY = tf.keras.utils.to_categorical(testy)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    #convert to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #norm to 0-1 range
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return train_norm, test_norm

def summerize_diagonstics(history):
    #plot loss
    plt.subplot(211)
    plt.title('Criss Entropy Loss')
    plt.plot(history.history['loss'], color ='blue', label ='train')
    plt.plot(history.history['val_loss'], color = 'orange', label = 'test')
    plt.legend()
    #plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color ='blue', label ='train')
    plt.plot(history.history['val_accuracy'], color = 'orange', label = 'test')
    plt.legend()
    filename = "Baseline 3 + test 3"
    plt.savefig(filename + '_plot.png')
    plt.close()
    plt.title('test')
    plt.plot(history.history['lr'], color ='blue', label ='lr')
    plt.show()

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return K.eval(optimizer.lr)
    return K.eval(lr)

def get_schedule(step):
    if step < 10:
        step = min(step, 10)
        return ((0.01 - 0.1) * (1 - step / 10) ** (1)) + 0.1
    else:
        step = min(step, 90)
        cosine_decay = 0.5 * (1 + cos(pi * step / 90))
        decayed = (1 - 0) * cosine_decay + 0
        return 0.1 * decayed
        
def define_model(funlist, adam):
    model = tf.keras.Sequential()
    for i in funlist:
        model.add(i)
    #compile
    lr_metiric = None
    if adam:
        opt = tf.keras.optimizers.Adam()
    else:
        opt = tf.keras.optimizers.SGD(momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

def conv2(i, L2= False, input=False):
    if input:
        if L2:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer=Ini, padding = 'same', kernel_regularizer=l2(0.001), input_shape=(32,32,3))
        else:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer=Ini, padding = 'same', input_shape=(32,32,3))
    else:
        if L2:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer=Ini, padding = 'same', kernel_regularizer=l2(0.001))
        else:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer=Ini, padding = 'same')

def max_pool():
    return tf.keras.layers.MaxPooling2D((2,2))

def flat():
    return tf.keras.layers.Flatten()

def dense(act, ini, i, L2=False):
    if ini:
        if L2:
            return tf.keras.layers.Dense(i, activation=act, kernel_initializer=Ini, kernel_regularizer=l2(0.001))
        else:
            return tf.keras.layers.Dense(i, activation=act, kernel_initializer=Ini)
    else:
        return tf.keras.layers.Dense(i, activation=act)
def drop(i=0.2):
    return tf.keras.layers.Dropout(i)

def augmentation(trainX, trainY, testX, testY, model, augment, schedule):
    if augment:
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow(trainX, trainY, batch_size= 64)
        steps = int(trainX.shape[0]/64)
        history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=100, validation_data=(testX, testY), callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule)], verbose=1)
    else:
        history = model.fit(trainX, trainY, epochs = 100, batch_size = 64, validation_data=(testX, testY), callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule)], verbose =1)
    return history

def batnorm():
    return tf.keras.layers.BatchNormalization()


def main():
    #TODO:
    # learning rate schedulers (lite svårare)

    print(tf.test.is_gpu_available())
    L2=False #Weightdecay
    aug= True #Dataaugmentation
    adam = False #adam optimizer
    #load data
    trainX, trainY, testX, testY = load_data()
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    #preprocess data
    trainX, testX = prep_pixels(trainX, testX)
    #define model
    scheduleLin = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.1, first_decay_steps = 50, t_mul = 1.0, m_mul= 1.0, alpha = 0.0)
    funlist = [conv2(1, L2, input = True), batnorm(), conv2(1, L2), batnorm(), max_pool(), drop(0.2), conv2(2, L2), batnorm(), conv2(2, L2), batnorm(), max_pool(),drop(0.3), conv2(4, L2), batnorm(), conv2(4, L2), batnorm(), max_pool(),drop(0.4), flat(), dense('relu', True, 128, L2), batnorm(), drop(0.5), dense('softmax', False, 10, L2)]
    model = define_model(funlist, adam)
    #fit model
    history = augmentation(trainX, trainY, testX, testY, model, aug, scheduleLin)
    #evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('>%.3f'%(acc*100.0))
    #learning curves
    summerize_diagonstics(history)

main()


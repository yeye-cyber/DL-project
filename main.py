
import sys
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

#för att kunna ladda ner datasetet om det inte funkar testa: pip install ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    filename = "Baseline 1"
    plt.savefig(filename + '_plot.png')
    plt.close()

def define_model(funlist):
    model = tf.keras.Sequential()
    for i in funlist:
        model.add(i)
    #compile
    opt = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def conv2(i, L2= False, input=False):
    if input:
        if L2:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer='he_uniform', padding = 'same', kernel_regularizer=l2(0.001), input_shape=(32,32,3))
        else:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer='he_uniform', padding = 'same', input_shape=(32,32,3))
    else:
        if L2:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer='he_uniform', padding = 'same', kernel_regularizer=l2(0.001))
        else:
            return tf.keras.layers.Conv2D(32*i, (3,3), activation='relu', kernel_initializer='he_uniform', padding = 'same')

def max_pool():
    return tf.keras.layers.MaxPooling2D((2,2))

def flat():
    return tf.keras.layers.Flatten()

def dense(act, ini, i, L2=False):
    if ini:
        if L2:
            return tf.keras.layers.Dense(i, activation=act, kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))
        else:
            return tf.keras.layers.Dense(i, activation=act, kernel_initializer='he_uniform')
    else:
        return tf.keras.layers.Dense(i, activation=act)
def drop(i=0.2):
    return tf.keras.layers.Dropout(i)

def augmentation(trainX, trainY, testX, testY, model, augment):
    if augment:
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow()
    else:
        history = model.fit(trainX, trainY, epochs = 100, batch_size = 64, validation_data=(testX, testY), verbose =1)


def main():
    print(tf.test.is_gpu_available())
    L2=False #Weightdecay
    #load data
    trainX, trainY, testX, testY = load_data()
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    #preprocess data
    trainX, testX = prep_pixels(trainX, testX)
    #define model
    funlist = [conv2(1, L2, input = True), conv2(1, L2), max_pool(), flat(), dense('relu', True, 128, L2), dense('softmax', False, 10, L2)]
    model = define_model(funlist)
    #fit model
    history = model.fit(trainX, trainY, epochs = 100, batch_size = 64, validation_data=(testX, testY), verbose =1)
    #evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('>%.3f'%(acc*100.0))
    #learning curves
    summerize_diagonstics(history)

main()


#import
from keras.applications import *
from keras.layers import *
from keras.models import *
from keras.activations import  *
from keras.datasets import *
from keras.utils import np_utils
from keras.optimizers import *

#dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print (x_train.shape)
print (x_train.shape[0])

#data_process
x_train=x_train.reshape(-1,28,28,1)/255.
x_test=x_test.reshape(-1,28,28,1)/255.
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

#model
model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),2))
model.add(Convolution2D(64,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=1024,validation_data=(x_test,y_test))
# loss,accuracy=model.evaluate(x_test,y_test,batch_size=1024,verbose=1)
# print(loss)
# print(accuracy)
prediction=model.predict(x_test)
print(prediction)


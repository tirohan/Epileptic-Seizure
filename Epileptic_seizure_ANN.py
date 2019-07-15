import numpy as np # linear algebra
import pandas as pd

import os
print(os.listdir("../input"))

dataset = pd.read_csv("../input/data.csv")
print(dataset.shape)

X = np.array(dataset.iloc[1:, 1:179].values)
y = np.array(dataset.iloc[1:, 179].values)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(y_train)


print(X_train.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 6)
y_test = np_utils.to_categorical(y_test, 6)
print(y_train.shape)

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(5000, input_shape=(178,)))
model.add(Activation('relu'))                            

          
model.add(Dense(2500))
model.add(Activation('relu'))

    
model.add(Dense(1000))
model.add(Activation('relu'))



model.add(Dense(600))
model.add(Activation('relu'))
          
model.add(Dense(2500))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(150))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))



model.add(Dense(78))
model.add(Activation('relu'))                            


#final......
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
#fit the model...

fit_model = model.fit(X_train, y_train,epochs=50)
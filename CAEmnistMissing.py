# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:30:20 2017

@author: rlguensa
"""

#import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

############################ Autoencoder preparation #######################

from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#####################" MNIST


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

############## create image with artificial missing data ############
nbTest=10
patchsize=28

Testimage=np.zeros((nbTest,patchsize*patchsize));
GroundTruth=np.zeros((nbTest,patchsize*patchsize))
OBSvec=np.zeros((nbTest,patchsize*patchsize))
for i in range(nbTest):             
    Testimage[i]=x_test[i,:].flatten().copy()
    GroundTruth[i]=x_test[i,:].flatten().copy()
    missRate=0.5
    missInd=np.nonzero(np.random.choice([0, 1], size=(Testimage.shape[1]), p=[1-missRate, missRate]))
    Testimage[i,missInd[0]]=float('nan')
    OBSvec[i]=Testimage[i].copy()

#########################  DINAE
ImageReconstructedResult=np.zeros((nbTest,patchsize*patchsize));

rmseFinal=[]

for ii in range(nbTest): 
    rmseVec=[];
    indMissingTest=np.where(np.isnan(Testimage[ii]))[0]
    Testimage[ii,indMissingTest]=0 #np.nanmean(Testimage); #or simply 0
    # iter 1
    iteration=1
    tempmax=np.amax(Testimage[ii])
    Testimage[ii]=Testimage[ii]/tempmax
    ImageReconstructed = autoencoder.predict(np.reshape(Testimage[None,ii], (1, 1, 28, 28)))
    Testimage[ii,indMissingTest]=(ImageReconstructed.flatten())[indMissingTest]
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    rmseError1 = sqrt(mean_squared_error(GroundTruth[ii],(ImageReconstructed.flatten())*tempmax))
    rmseVec.append(rmseError1)
    print rmseVec
    # next iterations
    rmseError=rmseError1
    OldrmseError=rmseError
    NewrmseError=0
    
    
    for j in range(1,100):
        OldrmseError=rmseError
        ImageReconstructed = autoencoder.predict(np.reshape(Testimage[None,ii], (1, 1, 28, 28))) 
        Testimage[ii,indMissingTest]=(ImageReconstructed.flatten())[indMissingTest];
        rmseError = sqrt(mean_squared_error(GroundTruth[ii],(ImageReconstructed.flatten())*tempmax))
        iteration=iteration+1
        NewrmseError=rmseError
        rmseVec.append(rmseError)
        if NewrmseError < OldrmseError:
            ImageReconstructedResult[ii,:]=(ImageReconstructed.flatten()).copy();
            print j
            continue
        else:
            break
        
    print rmseVec
    rmseFinal.append(rmseVec[-1])
    
print "rmsefinal is %f" %np.mean(rmseFinal)
################" reconstruction

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(GroundTruth[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display corrupted
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(OBSvec[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(ImageReconstructedResult[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#####################  HISTORY
## list all data in history
#print(history.history.keys())
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
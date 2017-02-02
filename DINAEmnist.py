

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:09:10 2017

@author: redouane lguensat

"""

from keras.layers import Input, Dense
from keras.models import Model


input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
#encoded = Dense(32, activation='relu')(encoded)

#decoded = Dense(64, activation='relu')(encoded)
#decoded = Dense(128, activation='relu')(encoded)  #!!!!!!
decoded = Dense(784, activation='sigmoid')(encoded)#!!!!!!

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape


autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


############## create image with artificial missing data ############
nbTest=10
patchsize=28

Testimage=np.zeros((nbTest,patchsize*patchsize));
GroundTruth=np.zeros((nbTest,patchsize*patchsize))
OBSvec=np.zeros((nbTest,patchsize*patchsize))
for i in range(nbTest):             
    Testimage[i]=x_test[i,:].copy()
    GroundTruth[i]=x_test[i,:].copy()
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
    ImageReconstructed = autoencoder.predict(Testimage[None,ii])
    Testimage[ii,indMissingTest]=ImageReconstructed[0,indMissingTest]
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    rmseError1 = sqrt(mean_squared_error(GroundTruth[ii],ImageReconstructed[0]*tempmax))
    rmseVec.append(rmseError1)
    print rmseVec
    # next iterations
    rmseError=rmseError1
    OldrmseError=rmseError
    NewrmseError=0
    
    
    for j in range(1,100):
        OldrmseError=rmseError
        ImageReconstructed = autoencoder.predict(Testimage[None,ii]) 
        Testimage[ii,indMissingTest]=ImageReconstructed[0,indMissingTest];
        rmseError = sqrt(mean_squared_error(GroundTruth[ii],ImageReconstructed[0]*tempmax))
        iteration=iteration+1
        NewrmseError=rmseError
        rmseVec.append(rmseError)
        if NewrmseError < OldrmseError:
            ImageReconstructedResult[ii,:]=ImageReconstructed[0].copy();
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

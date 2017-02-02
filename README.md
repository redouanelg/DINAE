# DINAE
DINAE for Data Interpolating AutoEncoders

This a code for a personal project (not published) on reconstructing misssing data using autoencoders. The idea behind DINAE is not new and it based on the following steps:

> Deep AE is trained using training set of complete images
> Image (or data) with missing data is passed to the input of the trained AE, where the missing values are replaced with 0.
> We get a first ReconstructedImage as an output and we take the new values of previously missing indices and put them back instead of the zeros
> Image (or data) with replaced values is again passed as an input to the AE
> We repeat until some stopping criterion

I put two examples using MNIST, one with deep AE where the image is vectorised (so the inputs are vectors of $28\times28$ length), the second script is more "realistic" for images where I used convolutional AEs. The codes are based on Keras.

If you have any remarks on the code (I'm still a Matlab guy, but I learn fast :D), or if you find some bugs do not hesitate to send me a message.

If you're a good guy (girl?) and desire to cite the code, please use the following:

@misc{DINAE, 
author = {Lguensat Redouane}, 
title = {Data Interpolating AutoEncoders}, 
year = {2016}, 
publisher = {GitHub}, 
journal = {GitHub repository}, 
howpublished = {\url{https://github.com/redouanelg/DINAE}} 
}

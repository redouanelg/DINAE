# DINAE
DINAE for Data Interpolating AutoEncoders

This a code for a personal project (not published) on reconstructing misssing data using autoencoders. The idea behind DINAE is not new (but haven't found a Python code with Keras implementing it) and is based on the following steps:

>* Deep AE is trained using training set of complete images <br />
>* Image (or data) with missing data is passed to the input of the trained AE, where the missing values are replaced with 0. <br />
>* We get a first ReconstructedImage as an output and we take the new values of previously missing indices and put them back instead of the zeros <br />
>* Image (or data) with replaced values is again passed as an input to the AE <br />
>* We repeat until some stopping criterion <br />

I put two examples using MNIST, one with deep AE where the image is vectorised (so the inputs are vectors of $28\times28$ length), the second script is more "realistic" for images where I used convolutional AEs. The codes are based on Keras.

If you have any remarks on the code, or if you find some bugs do not hesitate to send me a message.

![example](https://github.com/redouanelg/DINAE/blob/master/ae.png)

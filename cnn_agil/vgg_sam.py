'''
This code is part of the Keras VGG-16 model
Adopted from https://github.com/marcellacornia/sam/blob/master/dcn_vgg.py
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.utils.data_utils import get_file
from keras import backend as K

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'


def dcn_vgg(input_tensor=None):
    input_shape = (240,320,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv_1
    x = Convolution2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last", name='block1_pool')(x)
    print("conv_1", x.shape)

    # conv_2
    x = Convolution2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Convolution2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last", name='block2_pool')(x)
    print("conv_2", x.shape)

    # conv_3
    x = Convolution2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Convolution2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Convolution2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last", name='block3_pool', padding='same')(x)
    print("conv_3", x.shape)

    # conv_4
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), data_format="channels_last", name='block4_pool', padding='same')(x)
    print("conv_4", x.shape)

    # conv_5
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block5_conv1', dilation_rate=(2,2))(x)
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block5_conv2', dilation_rate=(2,2))(x)
    x = Convolution2D(512, (3,3), activation='relu', padding='same', name='block5_conv3', dilation_rate=(2,2))(x)
    print("conv_5", x.shape)
    
    
    # Deconv
    deconv12 = Conv2DTranspose(512, (2,2), strides=1, padding='same')
    x = deconv12(x)
    print(deconv12.output_shape)
    x=Activation('relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(0.5)(x)

    deconv22 = Conv2DTranspose(256, (2,2), strides=2, padding='same')
    x = deconv22(x)
    print(deconv22.output_shape)
    x=Activation('relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(0.5)(x)
    
    deconv23 = Conv2DTranspose(128, (2,2), strides=2, padding='same')
    x = deconv23(x)
    print(deconv23.output_shape)
    x=Activation('relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(0.5)(x)
    
    deconv24 = Conv2DTranspose(1, (2,2), strides=2, padding='same')
    x = deconv24(x)
    print(deconv24.output_shape)
    x=Activation('relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(0.5)(x)

    
    # Create model
    model = Model(img_input, x)

    # Load weights
    weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', TH_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    model.load_weights('./vgg16_weights.h5', by_name=True)

    return model

dcn_vgg()

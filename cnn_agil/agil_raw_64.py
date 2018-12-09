'''adopted from Arcade-Learning-Environment repository:
https://github.com/corgiTrax/Arcade-Learning-Environment/blob/master/gaze/main-salient-image%2Bopf-past4.py '''

import numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model
import tensorflow as tf 
import random
import os
from numpy.random import seed
from tensorflow import set_random_seed

import util_calculations as MU
import load_data_for_cnn
import base_misc_utils
import compress_data
import cv2

'''Fix random seed'''
#seed(0)
#set_random_seed(0)

os.environ['MKL_NUM_THREADS']='272'
os.environ['GOTO_NUM_THREADS']='272'
os.environ['OMP_NUM_THREADS']='272'
os.environ['openmp']='True'


print("Architecture: agil_bianzhong")


BATCH_SIZE = 64
validation_size = 200
validation_BATCH_SIZE = 50
num_epoch = 20

resume_model = False
predict_mode = False
compress_channel_1=True   # Whether to stack and save img by every 2 channels first (sub 1)
compress_channel_2 = True 
compress_channel_3 = True   

save_index = '61'   # File index to load the well-trained model and save predictions
dropout = 0.5
k = 3   # input img channels
stride = 1
SHAPE = (120,160,k) # height * width * channel This cannot read from file and needs to be provided here

print('predict?', predict_mode)

if not predict_mode: # If training, save in target dir
    expr = base_misc_utils.ExprCreaterAndResumer('result_sem2', \
						postfix="agil_dp%.1f_batch%d_chan%d_stride%d" % (dropout,BATCH_SIZE,k,stride))

base_misc_utils.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

# Load model
if resume_model:
    model = expr.load_weight_and_training_config_and_state()

# Write a new model
else:
    ###############################
    # Architecture of the network #
    ###############################
    # First channel: image

    x_inputs=L.Input(shape=SHAPE)    
    x=x_inputs #inputs is used by the line "Model(inputs, ... )" below

    # conv_1
    x = L.Conv2D(32, (4,4), strides=4, activation='relu', padding='valid', data_format="channels_last", name='block1_conv1')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    print("conv_1_1", x.shape)
    x = L.Conv2D(64, (4,4), strides=2, activation='relu', padding='valid', data_format="channels_last", name='block1_conv2')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    print("conv_1_2", x.shape)

    # conv_2
    x = L.Conv2D(64, (3,3), strides=1, activation='relu', padding='valid', data_format="channels_last", name='block2_conv1')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    print("conv_2", x.shape)
    
    # deconv
    deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
    x = deconv1(x)
    print("deconv_1", deconv1.output_shape)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
 
    deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
    x = deconv2(x)
    print("deconv_2", deconv2.output_shape)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x) 
    deconv3 = L.Conv2DTranspose(1, (4,4), strides=4, padding='valid')
    x = deconv3(x)
    print ("deconv_3", deconv3.output_shape)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    # Output layer
    x_output = L.Activation(MU.my_softmax)(x)    

    # Define model
    model=Model(inputs=x_inputs, outputs=x_output)

    # Load weights
    #model.load_weights('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/cnn/result_sem2/53_agil_grey_dp0.5_batch64_chan1_stride1/weights.20-4.41.hdf5', by_name=True)

    # Optimizer
    opt=K.optimizers.Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])

# Load training data and labels(heatmaps) 

if compress_channel_1:
    training_data_object = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_1/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_1/',
                                                   120, 160)

    sub1_data, sub1_labels, frame_index, heatmap_index = training_data_object.load_data()
    training_data_object2 = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_1_flip/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_1_flip/',
                                                   120, 160)

    sub1_data_flip, sub1_labels_flip, frame_index_flip, heatmap_index_flip = training_data_object2.load_data()

    #compress_data.save(k, './dir_to_save_compressed_frames_sub1_k6/', training_data, frame_index)

#training_data = compress_data.load_data(k, './dir_to_save_compressed_frames_sub1_k6/', 120, 160)
#training_labels = compress_data.load_label('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_1/',
 #                                                  120, 160)
print('sub1 data loaded, shape:', sub1_data.shape, sub1_labels.shape)

# Load test data and labels(heatmaps) 
if compress_channel_2:
    val_data_object = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_2/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_2/',
                                                       120, 160) 
    sub2_data, sub2_labels, frame_index_test, heatmap_index_test = val_data_object.load_data_validation()
    val_data_object2 = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_2_flip/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_2_flip/',
                                                   120, 160)

    sub2_data_flip, sub2_labels_flip, frame_index_test_flip, heatmap_index_test_flip = val_data_object2.load_data_validation()


#    compress_data.save(k, './dir_to_save_compressed_frames_sub2_k6/', test_data, frame_index_test)

#test_data = compress_data.load_data(k, './dir_to_save_compressed_frames_sub2_k6/', 120, 160)
#test_labels = compress_data.load_label('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_2/',
 #                                                  120, 160)
print('sub2 data loaded, shape:', sub2_data.shape, sub2_labels.shape)


# Load sub3 data and labels(heatmaps) 
if compress_channel_3:
    test_data_object = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_3/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_3/',
                                                       120, 160) 
    sub3_data, sub3_labels, frame_index_val, heatmap_index_val = test_data_object.load_data_test()
    test_data_object2 = load_data_for_cnn.read_data('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_3_flip/',
                                                   '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_3_flip/',
                                                   120, 160)

    sub3_data_flip, sub3_labels_flip, frame_index_val_flip, heatmap_index_val_flip = test_data_object2.load_data_test()

#    compress_data.save(k, './dir_to_save_compressed_frames_sub3_k12/', test_data, frame_index_test)

#val_data = compress_data.load_data(k, './dir_to_save_compressed_frames_sub3_k12/', 120, 160)
#val_labels = compress_data.load_label('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_3/',
#                                                   120, 160)
print('sub3 data loaded, shape:', sub3_data.shape, sub3_labels.shape)

# Combine subject data and labels
training_data, training_labels = np.vstack((sub1_data[0:3300], sub2_data[0:3800])), np.vstack((sub1_labels[0:3300], sub2_labels[0:3800]))
training_data, training_labels = np.vstack((training_data, sub3_data[0:3500])), np.vstack((training_labels, sub3_labels[0:3500]))

training_data, training_labels = np.vstack((training_data, sub1_data_flip[0:3300])), np.vstack((training_labels, sub1_labels_flip[0:3300]))
training_data, training_labels = np.vstack((training_data, sub2_data_flip[0:3800])), np.vstack((training_labels, sub2_labels_flip[0:3800]))
training_data, training_labels = np.vstack((training_data, sub3_data_flip[0:3500])), np.vstack((training_labels, sub3_labels_flip[0:3500]))
print('training set size:', training_data.shape, training_labels.shape)

test_data, test_labels = np.vstack((sub1_data[3300:], sub2_data[3800:])), np.vstack((sub1_labels[3300:], sub2_labels[3800:]))
test_data, test_labels = np.vstack((test_data, sub3_data[3500:])), np.vstack((test_labels, sub3_labels[3500:]))

print('test set size:', test_data.shape, test_labels.shape)

#std
mean = np.mean(training_data)
std = np.std(training_data)
training_data = (training_data - mean) / std
print('training mean&std:', mean, std)
test_data = (test_data - mean) / std

if not predict_mode: 

    # if training mode, save model and source codes
    expr.dump_src_code_and_model_def('/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/cnn/vgg_dec.py', model)
    
    #training_data_object.load_data(BATCH_SIZE = BATCH_SIZE, 
    #                               validation_size=validation_size,
    #                               validation_BATCH_SIZE=validation_BATCH_SIZE)
    # 
    model.fit(training_data, training_labels, BATCH_SIZE, epochs=num_epoch,
        validation_data=(test_data, test_labels),
        shuffle=True, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=num_epoch, min_lr = 0.00001),
            K.callbacks.ModelCheckpoint(expr.dir+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
            base_misc_utils.PrintLrCallback()])

   
    #model.fit_generator(training_data_object.batch_input(mode='training'), 
    #        steps_per_epoch=15,
    #        epochs=num_epoch,
    #        verbose=2,
    #        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
    #                   K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=num_epoch, min_lr = 0.00001),
    #                   base_misc_utils.PrintLrCallback()], use_multiprocessing=True, shuffle=True)
            #validation_data = training_data_object.batch_input(mode='validation'),
            #validation_steps = 3)

    expr.save_weight_and_training_config_state(model)
    expr.printdebug('Successfully saved')
    expr.logfile.close()

# if prediction mode
elif predict_mode:
    target = 'sub2'   # 'subx'
    rootdir = './result_sem2/' + str(save_index) + '_agil_dp0.5_batch'+ str(BATCH_SIZE) + '_chan'+ str(k) + '_stride1/'
    model.load_weights(rootdir +'weights.17-3.78.hdf5')

    #print ("Evaluating model...")
    #score = model.evaluate(test_data, test_labels, BATCH_SIZE, 0)
    #print ("Test score is: " , score)

    '''
    print ("Predicting training results...")
    if not os.path.exists(rootdir + 'prediction/sub1/'):
        os.makedirs(rootdir + 'prediction/sub1/')
    pred_training = model.predict(training_data, BATCH_SIZE)
    training_data = 0
    for i in range(pred_training.shape[0]):
        np.savez(rootdir + 'prediction/sub1/' + 'prediction%d' %i, heatmap=pred_training[i])

    print ("Predicted and saved(training).")'''

    print ("Copying truth frames...")

    training_data = 0
    test_data = 0   # Release memory

    if not os.path.exists(rootdir + 'prediction/' + target + '/'):   # File for prediction
        os.makedirs(rootdir + 'prediction/' + target + '/')
    if not os.path.exists(rootdir + 'prediction/groundtruth_frames_' + target + '/'):   # File for raw groundtruth frames
        os.makedirs(rootdir + 'prediction/groundtruth_frames_' + target + '/')

    # Copy ground truth frames for video visualization
    truth_dir = '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_groundtruth_' + target[-1] + '/'   
    #truth_dir = '/home1/05563/dl33629/WORK_maverick/gaze_prediction/Walking_Gaze_Modeling/frames_in_use_1/'
    truth_sets = [x for x in os.listdir(truth_dir) if x.endswith('.jpg')]
    # Sort file name by frame index
    for i in range(len(truth_sets)):
        truth_sets[i] = int(truth_sets[i].strip('frame').strip('.jpg'))
    truth_sets.sort()
    for i in range(len(truth_sets)):
         truth_sets[i] = 'frame' + str(truth_sets[i]) + '.jpg'

    # Save truth frames
    for i in range(len(truth_sets)):
          if i >= 3800:
              frame_idx = int(truth_sets[i].strip('frame').strip('.jpg'))
              img = cv2.imread(truth_dir + truth_sets[i])
              cv2.imwrite(rootdir + 'prediction/groundtruth_frames_' + target + '/' + 'frame%s.jpg' % frame_idx, img)

    # Predict
    print ("Predicting test results...")
    test_data = (sub2_data - mean) / std   # Standardization
    pred = model.predict(test_data[3800:], BATCH_SIZE) 
    for i in range(pred.shape[0]):
        np.savez(rootdir + 'prediction/' + target + '/' + 'prediction%d' %i, heatmap=pred[i])

    print ("Predicted and saved(test).") 

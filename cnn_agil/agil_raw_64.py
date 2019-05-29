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
seed(0)
set_random_seed(0)

os.environ['MKL_NUM_THREADS']='272'
os.environ['GOTO_NUM_THREADS']='272'
os.environ['OMP_NUM_THREADS']='272'
os.environ['openmp']='True'


print("Architecture: agil_bianzhong")


BATCH_SIZE = 64
validation_size = 200
validation_BATCH_SIZE = 50
num_epoch = 40

resume_model = False
predict_mode = True
load_sub1=True  
load_sub2 = True 
load_sub3 = True   

save_index = '71'   # File index to load the well-trained model and save predictions
dropout = 0.5
k = 3   # input img channels
stride = 1
SHAPE = (120,160,k) # height * width * channel This cannot read from file and needs to be provided here

sub1_data_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/crop_images_1/'
sub1_heatmap_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_1/'
sub2_data_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/crop_images_2/'
sub2_heatmap_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_2/'
sub3_data_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/crop_images_3/'
sub3_heatmap_file = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/groundtruth_heatmap_3/'

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
    #x = L.Dropout(dropout)(x)
    print("conv_1_1", x.shape)
    x = L.Conv2D(64, (4,4), strides=2, activation='relu', padding='valid', data_format="channels_last", name='block1_conv2')(x)
    x = L.BatchNormalization()(x)     
    #x = L.Dropout(dropout)(x)
    print("conv_1_2", x.shape)

    # conv_2
    x = L.Conv2D(64, (3,3), strides=1, activation='relu', padding='valid', data_format="channels_last", name='block2_conv1')(x)
    x = L.BatchNormalization()(x)
    #x = L.Dropout(dropout)(x)
    print("conv_2", x.shape)
    
    # deconv
    deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
    x = deconv1(x)
    print("deconv_1", deconv1.output_shape)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    #x = L.Dropout(dropout)(x)
 
    deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
    x = deconv2(x)
    print("deconv_2", deconv2.output_shape)
    x = L.Activation('relu')(x)
    x = L.BatchNormalization()(x)
    #x = L.Dropout(dropout)(x) 
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

''' Load sub1 data and labels(heatmaps) '''
if load_sub1:
    sub1_object = load_data_for_cnn.read_data(sub1_data_file, sub1_heatmap_file, SHAPE[0], SHAPE[1])

    sub1_data, sub1_labels, frame_index1, heatmap_index1 = sub1_object.load_data()
    #sub1_object = load_data_for_cnn.read_data(sub1_data_file, sub1_heatmap_file, SHAPE[0], SHAPE[1])
    #sub1_data_flip, sub1_labels_flip, frame_index1_flip, heatmap_index1_flip = sub1_object.load_data()

print('sub1 data loaded, shape:', sub1_data.shape, sub1_labels.shape)


''' Load sub2 data and labels(heatmaps) '''
if load_sub2:
    sub2_object = load_data_for_cnn.read_data(sub2_data_file, sub2_heatmap_file, SHAPE[0], SHAPE[1]) 
    sub2_data, sub2_labels, frame_index2, heatmap_index2 = sub2_object.load_data()
    #sub2_object = load_data_for_cnn.read_data(sub2_data_file, sub2_heatmap_file, SHAPE[0], SHAPE[1])
    #sub2_data_flip, sub2_labels_flip, frame_index2_flip, heatmap_index2_flip = sub2_object.load_data()

print('sub2 data loaded, shape:', sub2_data.shape, sub2_labels.shape)


''' Load sub3 data and labels(heatmaps) '''
if load_sub3:
    sub3_object = load_data_for_cnn.read_data(sub3_data_file, sub3_heatmap_file, SHAPE[0], SHAPE[1]) 
    sub3_data, sub3_labels, frame_index3, heatmap_index3 = sub3_object.load_data()
    #sub3_object = load_data_for_cnn.read_data(sub3_data_file, sub3_heatmap_file, SHAPE[0], SHAPE[1])
    #sub3_data_flip, sub3_labels_flip, frame_index3_flip, heatmap_index3_flip = sub3_object.load_data()

print('sub3 data loaded, shape:', sub3_data.shape, sub3_labels.shape)

# Import test idx, middle 1/3 of each subject is used for testing
test_idx_start_sub1, test_idx_end_sub1 = int(round(len(sub1_data)/3.0)), int(round(len(sub1_data)/3.0*2.0))
test_idx_start_sub2, test_idx_end_sub2 = int(round(len(sub2_data)/3.0)), int(round(len(sub2_data)/3.0*2.0))
test_idx_start_sub3, test_idx_end_sub3 = int(round(len(sub3_data)/3.0)), int(round(len(sub3_data)/3.0*2.0))
print('Test row index, sub1:', test_idx_start_sub1, test_idx_end_sub1)
print('Test row index, sub2:', test_idx_start_sub2, test_idx_end_sub2)
print('Test row index, sub3:', test_idx_start_sub3, test_idx_end_sub3)

# Combine subject data and labels
# Sub1
training_data, training_labels = np.vstack((sub1_data[0:test_idx_start_sub1], sub1_data[test_idx_end_sub1:])), np.vstack((sub1_labels[0:test_idx_start_sub1], sub1_labels[test_idx_end_sub1:]))
# Sub2
training_data, training_labels = np.vstack((training_data, sub2_data[0:test_idx_start_sub2])), np.vstack((training_labels, sub2_labels[0:test_idx_start_sub2]))
training_data, training_labels = np.vstack((training_data, sub2_data[test_idx_end_sub2:])), np.vstack((training_labels, sub2_labels[test_idx_end_sub2:]))
# Sub3
training_data, training_labels = np.vstack((training_data, sub3_data[0:test_idx_start_sub3])), np.vstack((training_labels, sub3_labels[0:test_idx_start_sub3]))
training_data, training_labels = np.vstack((training_data, sub3_data[test_idx_end_sub3:])), np.vstack((training_labels, sub3_labels[test_idx_end_sub3:]))
print('training set size:', training_data.shape, training_labels.shape)

# Form test set
test_data, test_labels = np.vstack((sub1_data[test_idx_start_sub1:test_idx_end_sub1], sub2_data[test_idx_start_sub2:test_idx_end_sub2])), np.vstack((sub1_labels[test_idx_start_sub1:test_idx_end_sub1], sub2_labels[test_idx_start_sub2:test_idx_end_sub2]))
test_data, test_labels = np.vstack((test_data, sub3_data[test_idx_start_sub3:test_idx_end_sub3])), np.vstack((test_labels, sub3_labels[test_idx_start_sub3:test_idx_end_sub3]))
print('test set size:', test_data.shape, test_labels.shape)

#std
mean = np.mean(training_data)
std = np.std(training_data)
training_data = (training_data - mean) / std
print('training mean&std:', mean, std)
test_data = (test_data - mean) / std

if not predict_mode: 

    # if training mode, save model and source codes
    expr.dump_src_code_and_model_def('/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/cnn/agil_raw_64.py', model)

    model.fit(training_data, training_labels, BATCH_SIZE, epochs=num_epoch,
        validation_data=(test_data, test_labels),
        shuffle=True, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=num_epoch, min_lr = 0.00001),
            K.callbacks.ModelCheckpoint(expr.dir+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
            base_misc_utils.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)
    expr.printdebug('Successfully saved')
    expr.logfile.close()

# if prediction mode
elif predict_mode:
    target = 'sub3'   # 'subx'
    data_to_test = sub3_data
    test_head, test_tail = test_idx_start_sub3, test_idx_end_sub3
    frame_index, heatmap_index = frame_index3, heatmap_index3
    rootdir = './result_sem2/' + str(save_index) + '_agil_dp0.5_batch'+ str(BATCH_SIZE) + '_chan'+ str(k) + '_stride1/'
    model.load_weights(rootdir +'weights.05-3.88.hdf5')  # 120*160:weights.03-3.67 ; 80*105:weights.04-4.00; 100*130:weights.05-3.88 ; 60*80:weights.06-4.15 ; 30*40:weights.05-4.25

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
    truth_dir = '/home1/05563/dl33629/WORK_stampede/gaze_prediction/Walking_Gaze_Modeling/frames_groundtruth_' + target[-1] + '/'   
    truth_sets = [x for x in os.listdir(truth_dir) if x.endswith('.jpg')]
    # Sort file name by frame index
    for i in range(len(truth_sets)):
        truth_sets[i] = int(truth_sets[i].strip('frame').strip('.jpg'))
    truth_sets.sort()
    for i in range(len(truth_sets)):
         truth_sets[i] = 'frame' + str(truth_sets[i]) + '.jpg'

    # Save truth frames, frame_index, data_to_test and truth_sets should align with each other exactly.
    print('Check if size of frame_index and data_to_test is the same:', len(frame_index), len(data_to_test))
    print('Check if an element of frame_index aligns with truth_sets:', frame_index[test_tail], truth_sets[test_tail])
    # Write truth frames
    for i in range(len(truth_sets)):
          frame_idx = int(truth_sets[i].strip('frame').strip('.jpg'))
          if i >= test_head and i < test_tail:
              img = cv2.imread(truth_dir + truth_sets[i])
              cv2.imwrite(rootdir + 'prediction/groundtruth_frames_' + target + '/' + 'frame%s.jpg' % frame_idx, img)

    # Predict
    print ("Predicting test results...")
    data_to_test = (data_to_test - mean) / std   # Standardization
    pred = model.predict(data_to_test[test_head:test_tail], BATCH_SIZE) 
    for i in range(len(pred)):
        np.savez(rootdir + 'prediction/' + target + '/' + 'prediction%d' %frame_index[test_head+i], heatmap=pred[i])

    print ("Predicted and saved(test).") 

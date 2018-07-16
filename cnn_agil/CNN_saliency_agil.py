'''adopted from Arcade-Learning-Environment repository:
https://github.com/corgiTrax/Arcade-Learning-Environment/blob/master/gaze/main-salient-image%2Bopf-past4.py '''

import numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model 


print("Usage: ipython main.py gamename [PredictMode?]")
print("Usage Predict Mode: ipython main.py gamename 1 parameters Model.hdf5")
print("Usage Training Mode: ipython main.py gamename 0 parameters")

#GAME_NAME = None
#if GAME_NAME == 'seaquest':
#    VAL_DATASET = ['75_RZ_3006069_Aug-17-16-46-05']
#    BASE_FILE_NAME = "G:/Research2/w5/"  
#    MODEL_DIR = "./"
# 
#
#LABELS_FILE_TRAIN = BASE_FILE_NAME + '-train.txt' 
#LABELS_FILE_VAL =  BASE_FILE_NAME + '-val.txt' 
#GAZE_POS_ASC_FILE = BASE_FILE_NAME + '.asc'
#AFFIX = '-image+opf_past4'
#PREDICT_FILE_VAL = BASE_FILE_NAME.split('/')[-1] + AFFIX
BATCH_SIZE = 50
num_epoch = 1

resume_model = False
predict_mode = True
dropout = 0.1
heatmap_shape = 84
k = 3
stride = 1
SHAPE = (576,1024,k) # height * width * channel This cannot read from file and needs to be provided here

#%%

import util_calculations as MU
import load_data_for_cnn
import base_misc_utils

if not predict_mode: # if train
    expr = base_misc_utils.ExprCreaterAndResumer('aaa', postfix="pKf_dp%.1f_k%ds%d" % (dropout,k,stride))
    expr.redirect_output_to_logfile_if_not_on("eldar-11")

base_misc_utils.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

# Load model
if resume_model:
    model = expr.load_weight_and_training_config_and_state()
    expr.printdebug("Checkpoint found. Resuming model at %s" % expr.dir_lasttime)
# Write a new model
else:
    ###############################
    # Architecture of the network #
    ###############################
    # First channel: image

    x_inputs=L.Input(shape=SHAPE)    
    x=x_inputs #inputs is used by the line "Model(inputs, ... )" below
    
    conv11=L.Conv2D(32, (8,8), strides=4, padding='same')
    x = conv11(x)
    x=L.Activation('relu')(x)
    x=L.BatchNormalization()(x)
    x=L.Dropout(dropout)(x)
    
#    conv12=L.Conv2D(64, (4,4), strides=2, padding='same')
#    x = conv12(x)
#    x=L.Activation('relu')(x)
#    x=L.BatchNormalization()(x)
#    x=L.Dropout(dropout)(x)
#    
#    conv13=L.Conv2D(64, (3,3), strides=1, padding='same')
#    x = conv13(x)
#    x=L.Activation('relu')(x)
#    x=L.BatchNormalization()(x)
#    x=L.Dropout(dropout)(x)
#    
#    deconv11 = L.Conv2DTranspose(64, (3,3), strides=1, padding='same')
#    x = deconv11(x)
#    x=L.Activation('relu')(x)
#    x=L.BatchNormalization()(x)
#    x=L.Dropout(dropout)(x)
#
#    deconv12 = L.Conv2DTranspose(32, (4,4), strides=2, padding='same')
#    x = deconv12(x)
#    x=L.Activation('relu')(x)
#    x=L.BatchNormalization()(x)
#    x=L.Dropout(dropout)(x)         

    deconv13 = L.Conv2DTranspose(3, (8,8), strides=4, padding='same')
    x = deconv13(x)
    print (deconv13.output_shape)
    x = L.Activation('relu')(x)
    x_output = L.BatchNormalization()(x)
    # Output layer
    outputs = L.Activation(MU.my_softmax)(x_output)
    # Define model
    model=Model(inputs=x_inputs, outputs=outputs)

    opt=K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opt=K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
    model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])

# Load training data and labels(heatmaps) 
training_data_object = load_data_for_cnn.read_data('G:/Research2/w5/frames_in_use/',
                                                   'G:/Research2/w5/rgb_saliency_maps/',
                                                   576, 1024)
training_data, training_labels = training_data_object.load_data()

# if training mode
if not predict_mode: 
    expr.dump_src_code_and_model_def('G:/Research2/w6/CNN_saliency_agil.py', model)

    model.fit(training_data, training_labels, BATCH_SIZE, epochs=num_epoch,
        validation_split=0.1,
        shuffle=True, verbose=2,
        callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 0.00001),
            base_misc_utils.PrintLrCallback()])

    expr.save_weight_and_training_config_state(model)
    expr.printdebug('Successfully saved')
    expr.logfile.close()

# if prediction mode
elif predict_mode: 
    model.load_weights('G:/Research2/w6/aaa/17_pKf_dp0.1_k3s1/'+'model.hdf5')

    print ("Evaluating model...")
    #train_score = model.evaluate([d.train_imgs, of.train_flow], d.train_GHmap, BATCH_SIZE, 0, sample_weight=d.train_weight)
    #print "Train loss is:  " , train_score
    val_score = model.evaluate(training_data, training_labels, BATCH_SIZE, 0)
    print ("Val loss is: " , val_score)

    print ("Predicting results...")
    train_pred = model.predict(training_data, BATCH_SIZE) 
    val_pred = model.predict(training_data, BATCH_SIZE)
    print ("Predicted.")


    # Uncomment this block to save predicted gaze heatmap for visualization
    #print "Converting predicted results into png files and save..."
    #IU.save_heatmap_png_files(d.train_fid, train_pred, TRAIN_DATASET, '../../saliency/')
    #IU.save_heatmap_png_files(d.val_fid, val_pred, VAL_DATASET, '../../saliency/')
    #print "Done."
#%%
#    print ("Writing predicted gaze heatmap (train) into the npz file...")
#    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX, fid=d.train_fid, heatmap=train_pred)
#    print ("Done. Output is:")
#    print (" %s" % BASE_FILE_NAME.split('/')[-1] + '-train' + AFFIX + '.npz')
#
#    print ("Writing predicted gaze heatmap (val) into the npz file...")
#    np.savez_compressed(BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX, fid=d.val_fid, heatmap=val_pred)
#    print ("Done. Output is:")
#    print (" %s" % BASE_FILE_NAME.split('/')[-1] + '-val' + AFFIX + '.npz')

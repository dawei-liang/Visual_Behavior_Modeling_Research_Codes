# Walking_Gaze_Modeling

To extract frames from the video, use _jon_data_2_frames.py_ (_check_dirs.py_)

To obtain frames in use: _label_useful_frames.py_ (_config.py, check_dirs.py_)

To generate ground truth marked frames, gaze log file and video for validation and to generate ground truth heatmap for training/testing, use _validation.py_ (_config.py, video.py, heatmap.py, check_dirs.py_)
(the data of subject 2 was stored in mat format, please use _validation_sub2.py_ instead)

Base line model(Itti, chance, anti_saliency): _main.py_

DNN models: _VGG.py_ (_util_calculations.py, load_data_for_cnn.py, base_misc_utils.py_)

To replay predictions: _replay_video.py_ (_check_dirs.py_)

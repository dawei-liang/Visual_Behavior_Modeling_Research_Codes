# Walking_Gaze_Modeling

To extract frames from the video, use _jon_data_2_frames.py_ (_check_dirs.py_)

To generate ground truth marked frames, gaze log file and video for validation and to generate ground truth heatmap for training/testing, use _validation.py_ (_config.py, video.py, heatmap.py, check_dirs.py_)
(the data of subject 2 was stored in mat format, please use _validation_sub2.py_ instead)

Base line model(Itti, chance, anti_saliency): _main.py_

Parameters: _config.py_

To get copies of in use frames: _label_useful_frames.py_

To replay predictions: _replay_video.py_

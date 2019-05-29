# Walking_Gaze_Modeling

To extract frames from the video, use _jon_data_2_frames.py_ (_check_dirs.py_)

To obtain frames in use: _label_useful_frames.py_ (_config.py, check_dirs.py_)

To generate cropped image frames, ground truth heatmap(cropped), gaze log files(cropped), frames with plotted gaze points and a video for visualization, please use _validation.py_ (_config.py, video.py, heatmap.py, check_dirs.py, randomly_crop_image.py, read_frameInfo.py, resize_patch_for_cnn.py_)

To flip img/heatmap/ground truth: _flip_img.py_(_check_dirs.py, config.py_)

Base line model(Itti, chance, anti_saliency): _main.py_ (_config.py, check_dirs.py_, _itti_model.py_, _heatmap.py_, _metrics.py_, _pySaliencyMap.py_, _pySaliencyMapDefs.py_)

DNN models: _VGG.py_ (_util_calculations.py, load_data_for_cnn.py, base_misc_utils.py_)

To replay predictions: _replay_video.py_ (_check_dirs.py_)

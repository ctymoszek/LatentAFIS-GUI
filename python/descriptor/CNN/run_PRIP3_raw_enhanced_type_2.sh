python src/classifier_train.py --gpu 1 --patch_type 2 --batch_size 128 --logs_base_dir /research/prip-kaicao/AutomatedLatentRecognition/log_descriptor/mobilenet_patch_type_2/ \
--models_base_dir /research/prip-kaicao/AutomatedLatentRecognition/models/mobilenet_patch_type_2/ \
--data_dir /research/prip-kaicao/LatentSearchData/Descriptor/TrainingMinutiaePatch_1131_subfolders/:/research/prip-kaicao/LatentSearchData/Descriptor/TrainingMinutiaePatch_Enh_1131_subfolders/ \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt

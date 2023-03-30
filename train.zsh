python RAFT256-PIV_train.py --nodes 1 --gpus 1 --name RAFT256-PIV_Onecycle_ProbClass2 \
--batch_size 10 --epochs 800 --recover False --output_dir_ckpt ./results/ \
--train_tfrecord ../RAFT-SS-DEV/data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
--train_tfrecord_idx ../RAFT-SS-DEV/data/idx_files/training_dataset_ProbClass2_256px.idx \
--val_tfrecord ../RAFT-SS-DEV/data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
--val_tfrecord_idx ../RAFT-SS-DEV/data/idx_files/validation_dataset_ProbClass2_256px.idx \
--log_path './train.log'
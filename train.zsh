# python RAFT256-PIV_train.py --nodes 1 --gpus 1 --name RAFT256-PIV_newModel_ProbClass2 \
# --batch_size 10 --epochs 50 --output_dir_ckpt ./results/ \
# --train_tfrecord ./data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --train_tfrecord_idx ./data/idx_files/training_dataset_ProbClass2_256px.idx \
# --val_tfrecord ./data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --val_tfrecord_idx ./data/idx_files/validation_dataset_ProbClass2_256px.idx \
# --log_path './train.log' \
# --recover True --input_path_ckpt './results/RAFT256-PIV_newModel_ProbClass2/ckpt.tar' \
# --init_lr 0.0001 --patience_level 5

# python RAFT256-PIV_train.py --nodes 1 --gpus 1 --name RAFT256-PIV_Onecycle_ProbClass2 \
# --batch_size 10 --epochs 50 --recover False --output_dir_ckpt ./results/ \
# --train_tfrecord ./data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --train_tfrecord_idx ./data/idx_files/training_dataset_ProbClass2_256px.idx \
# --val_tfrecord ./data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --val_tfrecord_idx ./data/idx_files/validation_dataset_ProbClass2_256px.idx \
# --log_path './train.log' --lr 0.0001

# RAFT-SS
# python RAFT256-PIV_train_copy.py --nodes 1 --gpus 1 --name RAFT256-PIV_SS_ProbClass2 \
# --batch_size 10 --epochs 50 --output_dir_ckpt ./results/ \
# --train_tfrecord ./data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --train_tfrecord_idx ./data/idx_files/training_dataset_ProbClass2_256px.idx \
# --val_tfrecord ./data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
# --val_tfrecord_idx ./data/idx_files/validation_dataset_ProbClass2_256px.idx \
# --log_path './train.log' \
# --init_lr 0.0001 --patience_level 5 \
# --recover True --input_path_ckpt 'results/RAFT256-PIV_SS_ProbClass2/ckpt.tar'

python RAFT256-SS-PIV_train.py --nodes 1 --gpus 1 --name RAFT256-PIV_SS_ProbClass2_2 \
--batch_size 10 --epochs 50 --output_dir_ckpt ./results/ \
--train_tfrecord ./data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
--train_tfrecord_idx ./data/idx_files/training_dataset_ProbClass2_256px.idx \
--val_tfrecord ./data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001 \
--val_tfrecord_idx ./data/idx_files/validation_dataset_ProbClass2_256px.idx \
--log_path './train.log' \
--init_lr 0.0000008 --patience_level 5 \
--recover True --input_path_ckpt 'results/RAFT256-PIV_SS_ProbClass2/ckpt.tar'
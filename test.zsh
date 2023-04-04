# python RAFT256-PIV_test.py --nodes 1 --gpus 1 --name RAFT256-PIV_test_cylinder \
# --input_path_ckpt ./results/ckpt.tar --test_dataset cylinder \
# --plot_results False --output_dir_results ./results/

# python RAFT256-PIV_test.py --nodes 1 --gpus 1 --name RAFT256-PIV_test_cylinder \
# --input_path_ckpt ./results/ckpt.tar --test_dataset backstep \
# --plot_results False --output_dir_results ./results/

# python RAFT256-PIV_test.py --nodes 1 --gpus 1 --name RAFT256-PIV_test_cylinder1 \
# --input_path_ckpt ./results/RAFT256-PIV_newModel_ProbClass2/ckpt.tar --test_dataset cylinder \
# --plot_results False --output_dir_results ./results/

python RAFT256-PIV_test.py --nodes 1 --gpus 1 --name RAFT256-PIV_test_cylinder1 \
--input_path_ckpt ./results/RAFT256-PIV_newModel_ProbClass2/ckpt.tar --test_dataset sqg \
--plot_results False --output_dir_results ./results/
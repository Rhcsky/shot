#tmux new-session -d -s 1test
#tmux send-keys -t 1test "CUDA_VISIBLE_DEVICES=4 python train_target.py train=target dataset=office-home dataset.t=1" Enter

#tmux new-session -d -s shot
#tmux send-keys -t shot "CUDA_VISIBLE_DEVICES=4 python train_target.py train=target dataset=RMFD" Enter

#tmux new-session -d -s 2shot
#tmux send-keys -t 2shot "CUDA_VISIBLE_DEVICES=5 python train_source.py train=source dataset=RMFD name=pretrained_RMFD" Enter

#tmux new-session -d -s shot
#tmux send-keys -t shot "CUDA_VISIBLE_DEVICES=5 python train_target.py train=target dataset=RMFD name=RMFD train.saved_model_path=../train_source_RMFD_pda_08-16_20-07 train.use_pretrained_backbone=False" Enter

tmux new-session -d -s targetshot
tmux send-keys -t targetshot "CUDA_VISIBLE_DEVICES=6 python train_target.py train=target dataset=RMFD name=pretrained_RMFD train.saved_model_path=../train_source_RMFD_pda_08-17_11-41" Enter
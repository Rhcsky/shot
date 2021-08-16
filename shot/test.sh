#tmux new-session -d -s 1test
#tmux send-keys -t 1test "CUDA_VISIBLE_DEVICES=4 python train_target.py train=target dataset=office-home dataset.t=1" Enter

#tmux new-session -d -s shot
#tmux send-keys -t shot "CUDA_VISIBLE_DEVICES=4 python train_target.py train=target dataset=RMFD" Enter

tmux new-session -d -s 2shot
tmux send-keys -t 2shot "CUDA_VISIBLE_DEVICES=5 python train_source.py train=source dataset=RMFD name=pretrained_RMFD" Enter


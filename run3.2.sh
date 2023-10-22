export CUBLAS_WORKSPACE_CONFIG=:16:8

python3 AI6103-HomeworkAssignment/main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 15 \
--lr 0.05 --wd 0 \
--mixup \
--seed 0 \
--fig_name 3.2-lr=0.05-lr_sche-wd=0-mixup.png \
--test


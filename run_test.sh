export CUBLAS_WORKSPACE_CONFIG=:16:8

python3 AI6103-HomeworkAssignment/main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.05 --wd 0.0005 \
--lr_scheduler \
--alpha 0.2 \
--mixup \
--seed 0 \
--fig_name 6.1-lr=0.05-lr_sche=CALR-wd=0.0005-mixup=0.2.png \
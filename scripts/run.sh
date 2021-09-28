# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
#python -W ignore ../train_supervised.py  --dataset CIFAR-FS --trial pretrain --model_path ../save/model/ckpt --data_root ../save/data/CIFAR-FS --use_trainval

# distillation
# setting '-a 1.0' should give simimlar performance
#python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
#python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root

python -W ignore ../train_distillation.py -r 0.5 -a 0.5 --dataset CIFAR-FS \
  --path_t ../save/model/ckpt/resnet12_CIFAR-FS_lr_0.05_decay_0.0005_trans_D_trial_pretrain_trainval/resnet12_last.pth \
  --trial newborn1 \
  --model_path ../save/model/ckpt \
  --data_root ../save/data/CIFAR-FS \
  --distill contrast \




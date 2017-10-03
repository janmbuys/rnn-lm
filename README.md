# RNN Langauge model

Sentence-level RNN language modelling, optimized for PTB.
Batches consist of sentences with exactly the same length.
Pre-processing implements Berkeley parser UNK classes.

Pytorch and python 3.

Usage example:

    python language_model.py --data_dir ptb/ --data_working_dir ptb-working/ --working_dir working --cuda --logging_interval 1000 --epochs 80 --embedding_size 650 --hidden_size 650 --lr 1.0 --dropout 0.5 --num_init_lr_epochs 6 --lr_decay 1.2 --init_weight_range 0.05 --grad_clip 5 --batch_size 16 --num_layers 2 


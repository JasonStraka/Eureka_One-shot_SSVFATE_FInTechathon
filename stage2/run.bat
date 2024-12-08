stage 2:
pip install pillow
python main_pretrain.py --dataset ctr_avazu2party --model dnnfm --input_size 32 --batch_size 512 --k 2 --pretrain_method moco --local_ssl 1 --aggregation_mode pma --aligned_label_percent 0.4 --seed 0

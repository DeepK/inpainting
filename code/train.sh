python3 train_3dunet.py ../models/vanilla_unet/ 150 0.0001 0
python3 train_3dunet.py ../models/finetuned_roi_superpixel_unet1/ 150 0.00005 1
python3 train_3dunet.py ../models/finetuned_roi_superpixel_unet2/ 150 0.0001 1
python3 train_3dunet.py ../models/finetuned_noroi_superpixel_unet1/ 150 0.00005 2
python3 train_3dunet.py ../models/finetuned_noroi_superpixel_unet2/ 150 0.0001 2
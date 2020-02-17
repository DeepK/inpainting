python3 train_3dunet.py ../models/vanilla_unet/ 150 0.0001 0
python3 train_3dunet.py ../models/finetuned_roi_superpixel_unet/ 150 0.0001 1
python3 train_3dunet.py ../models/finetuned_noroi_superpixel_unet/ 150 0.0001 2
python3 train_3dunet.py ../models/finetuned_roi_grid_unet/ 150 0.0001 3
python3 train_3dunet.py ../models/finetuned_noroi_grid_unet/ 150 0.0001 4
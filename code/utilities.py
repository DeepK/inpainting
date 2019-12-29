import os
from sklearn.model_selection import train_test_split

def windowed_scan_gen_xaxis(img, x_window_dim, extra = None):
    """
    Split the image row-wise (horizontally)
    This works for BRATS data as the brain is roughly symmetric and therefore the
    split can be made horizontally. Needs rewriting if we need to make
    this windowing generic
    """
    start_x = 0
    while start_x < img.shape[0]:
        if extra is None:
            yield img[start_x: min(start_x + x_window_dim, img.shape[0])]
        else:
            yield (img[start_x: min(start_x + x_window_dim, img.shape[0])],\
                extra[start_x: min(start_x + x_window_dim, img.shape[0])])
        start_x = start_x + x_window_dim

def get_flair_file_names(parent_brats_dir):
    names = os.listdir(parent_brats_dir)
    flair_names = [n + "/" + n + "_flair.nii.gz" for n in names]
    seg_names = [n + "/" + n + "_seg.nii.gz" for n in names]
    return flair_names, seg_names

def traintestvalid_split5(flair_names, seg_names):
    "5 different train-test splits"
    seeds = [1, 2, 3, 4, 5]

    train_test_splits = {}
    for seed in seeds:
        train_flair_names, test_flair_names, train_seg_names, test_seg_names = train_test_split(flair_names, seg_names, test_size=0.15, random_state=seed)
        train_flair_names, valid_flair_names, train_seg_names, valid_seg_names = train_test_split(train_flair_names, train_seg_names, test_size=0.15, random_state=seed)
        train_test_splits[seed] = (train_flair_names,\
                                   test_flair_names,\
                                   valid_flair_names,\
                                   train_seg_names,\
                                   test_seg_names,\
                                   valid_seg_names)

    return train_test_splits

if __name__ == "__main__":
    from reader import get_scan
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
    img = get_scan(fpath)
    
    from matplotlib import pyplot as plt
    
    img = img[40:200, 40:200, :]
    
    for img_tmp in windowed_scan_gen_xaxis(img, 80):
        for i in range(10, 140):
            fig = plt.figure("image")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img_tmp[:, :, i], interpolation = "nearest")
            plt.axis("off")
            plt.show()

    brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_test_splits = traintestvalid_split5(flair_names, seg_names)
    print (len(train_test_splits[1][0]),\
           len(train_test_splits[1][1]),\
           len(train_test_splits[1][2]),\
           len(train_test_splits[1][3]),\
           len(train_test_splits[1][4]),\
           len(train_test_splits[1][5]))
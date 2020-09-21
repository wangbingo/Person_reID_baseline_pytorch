import os
from shutil import copyfile
from IPython import embed

# You only need to change this line to your dataset download path
download_path = '../train'


#---------------------------------------
#train_val
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'

""" if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

os.system('cp -r ../train/pytorch/train_all/* ../train/pytorch/train/')  # tested ok.
 """

for root, dirs, files in os.walk(train_save_path, topdown=True):
    for dir in dirs:
        embed()
        """ os.mkdir(val_save_path + '/' + dir)


        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
 """
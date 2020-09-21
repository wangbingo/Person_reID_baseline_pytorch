import os, random, shutil
from shutil import copyfile
from IPython import embed

# You only need to change this line to your dataset download path
download_path = '../train'


#---------------------------------------
#train_val

train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

print('cp train_all train begin........')
os.system('cp -r ../train/pytorch/train_all/* ../train/pytorch/train/')  # tested ok.
print('cp train_all train completed........')

split_rate = 0.1

dir_numbers = len(os.listdir(train_save_path))    # 19658
pick_numbers = int(dir_numbers * split_rate)      #
dir_samples = random.sample(os.listdir(train_save_path), pick_numbers)  #

for dir in dir_samples:
    shutil.move(train_save_path + '/' + dir, val_save_path + '/' + dir)

print('{} / {} dirs moved from train to val.'.format(len(dir_samples), dir_numbers))
print('train/val datasets generated.')

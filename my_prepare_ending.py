import os, random, shutil
from shutil import copyfile
from IPython import embed

# You only need to change this line to your dataset download path
download_path = '../train'


#---------------------------------------
#train_val
def moveFile(srcDir, dstDir, rate = 0.1):
    pathDir = os.listdir(srcDir)    #取图片的原始路径
    filenumber=len(pathDir)
    if filenumber <= 3:  return   # train : val = 4 : 1
    #rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=max(1, int(filenumber * rate)) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    os.mkdir(dstDir)
    for name in sample:
            shutil.move(srcDir + '/' + name, dstDir + '/' + name)
    return

train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

print('cp train_all train begin........')
os.system('cp -r ../train/pytorch/train_all/* ../train/pytorch/train/')  # tested ok.
print('cp train_all train completed........')

split_rate = 0.1
#for root, dirs, files in os.walk(train_save_path, topdown=True):
#    for dir in dirs:

dir_numbers = len(os.listdir(train_save_path))    # 19658
pick_numbers = int(dir_numbers * split_rate)      #按照rate比例从dir中取一定数量dir
dir_samples = random.sample(os.listdir(train_save_path), pick_numbers)  #随机选取picknumber数量的dir

for dir in dir_samples:
    shutil.move(train_save_path + '/' + dir, val_save_path + '/' + dir)
    
print('{} / {} dirs moved.'.format(len(dir_samples), dir_numbers)
print('train/val  datasets completed.  split rate is {}'.format(split_rate))
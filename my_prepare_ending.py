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
    if filenumber <= 5:  return
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

os.system('cp -r ../train/pytorch/train_all/* ../train/pytorch/train/')  # tested ok.

split_rate = 0.1
c = 0
for root, dirs, files in os.walk(train_save_path, topdown=True):
    for dir in dirs:
        # os.mkdir(val_save_path + '/' + dir)
        moveFile(train_save_path + '/' + dir, val_save_path + '/' + dir, split_rate)
        c += 1
        if c % 2000 == 0:
            print('{} dirs processed.'.format(c))
print('train/val  datasets completed.  split rate is {}'.format(split_rate))
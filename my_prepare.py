import os, random, shutil
from shutil import copyfile
from IPython import embed

# You only need to change this line to your dataset download path
download_path = '../train'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
""" query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0] 
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
 """
#-----------------------------------------
#multi-query
""" query_path = download_path + '/gt_bbox'
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
 """
#-----------------------------------------
#gallery
""" gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
 """
#---------------------------------------
#train_all
train_path = download_path
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

file_handle = open(train_path + "/label.txt", "r") 
lines = file_handle.readlines()
file_handle.close()

print("There are {} lines in label.txt. Is that 72824?".format(len(lines)))

c = 0
for line in lines:
    line = line.strip()   # to erase blank
    line = line.strip('\n')  # to erase \n
    line_list = line.split(':')
    
    img_name  = line_list[0]
    cls_name = line_list[1] 
    
    src_path = train_path + '/images/' + img_name
    dst_path = train_save_path + '/' + cls_name
    
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + img_name)
    c += 1
    if c % 10000 == 0:
        print('{} files copied.'.format(c)) 

print('train_all dataset completed.  {} files copied.'.format(c))


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

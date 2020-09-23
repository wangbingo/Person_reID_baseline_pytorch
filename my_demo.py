import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../train/pytorch',type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    return index

result_dict = {}

q_index = opts.query_index  # from 1~2900

for i in range(q_index):

    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

    query_path, _ = image_datasets['query'].imgs[i]
    #     query_path = '../train/pytorch/query/11/00002570.png'
    query_path = query_path.split('/')[-1] # get '00002570.png'

    img_path_list = []
    for j in range(200):
        img_path, _ = image_datasets['gallery'].imgs[index[j]]
        #       img_path = '../train/pytorch/gallery/99/00108716.png'
        img_path = img_path.split('/')[-1]       # get '00108716.png'
        img_path_list.append(img_path)

    result_dict[query_path] = img_path_list
    if i % 100 == 0:
        print('{}/{} processed..........'.format(i+1, q_index))

import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

with open('result_' + str(nowTime) + '.json','w') as fp:
    json.dump(result_dict, fp, indent = 4, separators=(',', ': '))
    # json.dump(result_dict, fp)

print('The result generated................')
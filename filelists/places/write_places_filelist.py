import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd()
data_path = join(cwd,'source/places365_standard/train')
savedir = './'
dataset_list = ['base','val','novel']


folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    cfs = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
    cfs.sort()
    cfs = cfs[:200]
    classfile_list_all.append([ join(folder_path, cf) for cf in cfs])
    random.shuffle(classfile_list_all[i])
print(len(classfile_list_all))

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)

base = json.load(open('base.json', 'r'))
val = json.load(open('val.json', 'r'))
novel = json.load(open('novel.json', 'r'))

fo = open(savedir + "all.json", "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item for item in base['label_names']])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item for item in base['image_names']+val['image_names']+novel['image_names']])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item for item in base['image_labels']+val['image_labels']+novel['image_labels']])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("all -OK")
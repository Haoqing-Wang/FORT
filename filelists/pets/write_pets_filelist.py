from os.path import join
import os
import joblib

cwd = os.getcwd()
data_path = join(cwd, 'images')
savedir = './'

n_classes = 37
class_list = [str(i) for i in range(n_classes)]
image_to_label = joblib.load('image_to_label.pkl')
file_list = [join(data_path, k+'.jpg') for k, v in image_to_label.items()]
label_list = [int(v) for k, v in image_to_label.items()]

fo = open(savedir + "all.json", "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item for item in class_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item for item in file_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item for item in label_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("all -OK")
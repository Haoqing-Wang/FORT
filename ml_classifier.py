import os
import torch
import datetime
import time
from pathlib import Path

import methods.backbone_vit as backbone_vit
import methods.backbone_swin as backbone_swin
from dataset import SetDataManager
from options import parse_args

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

import random
import numpy as np
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def evaluate(novel_loader, n_way=5, n_support=5, n_query=15):
    # Model
    if 'vit' in params.backbone:
        encoder = backbone_vit.__dict__[params.backbone]().cuda()
    elif 'swin' in params.backbone:
        encoder = backbone_swin.__dict__[params.backbone]().cuda()
    else:
        print('Unknown backbone!')
        return
    for p in encoder.parameters():
        p.requires_grad = False

    # Update model
    checkpoint = torch.load(os.path.join('./Pretrain', params.pretrain), map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    msg = encoder.load_state_dict(checkpoint, strict=False)
    print(msg)
    encoder.eval()

    # test
    iter_num = len(novel_loader)
    acc_all = []
    ys = np.repeat(range(n_way), n_support)
    yq = np.repeat(range(n_way), n_query)
    test_log_file = open(os.path.join(params.output, '%s_%dw_%ds.txt' % (params.method, params.n_way, params.n_shot)), "w")
    start_time = time.time()
    for ti, (x, _) in enumerate(novel_loader):
        x = x.cuda()  # (5, 20, 3, 224, 224)
        with torch.no_grad():
            features, _ = encoder(x.reshape(-1, *x.size()[2:]))
            features = features.cpu().numpy()
        features = features.reshape(n_way, n_support+n_query, -1)
        xs = features[:, :n_support].reshape(n_way*n_support, -1)
        xq = features[:, n_support:].reshape(n_way*n_query, -1)

        if params.method == 'NN':
            classifier = KNeighborsClassifier(n_neighbors=1)
        elif params.method == 'RR':
            classifier = RidgeClassifier(alpha=1., fit_intercept=True)
        elif params.method == 'SVM':
            classifier = SVC(C=1., kernel='rbf')
        else:
            print("Please specify the method!")
            return False
        classifier.fit(xs, ys)
        acc = classifier.score(xq, yq) * 100.
        acc_all.append(acc)
        print('Task %d: %4.2f%%' % (ti, acc))
        print("Task %d: %4.2f%%" % (ti, acc), file=test_log_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
    print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96*acc_std/np.sqrt(iter_num)), file=test_log_file)

    print('Total time: {}'.format(total_time_str))
    print('Total time: {}'.format(total_time_str), file=test_log_file)
    test_log_file.close()


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args()
    print(params)

    image_size = 224
    eposide_num = 2000
    n_query = 15

    print('Loading target dataset!')
    novel_file = os.path.join(params.data_dir, params.dataset, 'all.json')
    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.n_way, n_support=params.n_shot, n_eposide=eposide_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    params.output = os.path.join(params.output, params.dataset)
    Path(params.output).mkdir(parents=True, exist_ok=True)
    evaluate(novel_loader, n_way=params.n_way, n_support=params.n_shot, n_query=n_query)
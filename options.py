import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot script')
    parser.add_argument('--dataset', default='cub', help='cub/cars/places/plantae/aircraft/pets')
    parser.add_argument('--backbone', default='vit_base_patch16', help='vit_base_patch16/swin_tiny')
    parser.add_argument('--method', default='RR', help='NN/RR/SVM/ProtoNet/R2D2/MetaOptNet')
    parser.add_argument('--n_way', default=20, type=int, help='class number to classify for evaluation')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class')
    parser.add_argument('--data_dir', default='./filelists', type=str, help='data path')
    parser.add_argument('--pretrain', default='dino_vit_base.pth', type=str, help='pre-trained model')
    parser.add_argument('--output', default='./output', type=str, help='')

    # Finetune
    parser.add_argument('--ft_epoch', default=5, type=int, help='')
    parser.add_argument('--ft_lr', default=1e-3, type=float, help='')
    parser.add_argument('--ft_wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--reset_head', default=False, type=bool, help='whether to use classifier initialization')

    parser.add_argument('--alpha', default=0., type=float, help='')
    parser.add_argument('--tau', default=1., type=float, help='')
    parser.add_argument('--lamb', default=1., type=float, help='')
    parser.add_argument('--P', default=14, type=int, help='')
    return parser.parse_args()
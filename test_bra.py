import argparse
import os
import torch
import shutil
from glob import glob
from networks.net_factory import net_factory
import torch
import torch.nn as nn
from test_bra_util import test_all_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/opt/data/private/yzy/data/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ALHVR', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_fea_aux', help='model_name')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
args = parser.parse_args()

def Inference(FLAGS):
    snapshot_path = "/opt/data/private/yzy/train_alhvr/code/model/Brats_{}_25_labeled/".format(FLAGS.exp)
    num_classes = 2
    test_save_path = "/opt/data/private/yzy/train_alhvr/code/model/Brats_{}_25_labeled/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type="vnet_fea_aux", in_chns=1, class_num=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(net).to(device)

    save_mode_path = os.path.join(
           snapshot_path, 'vnet_fea_aux_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric = Inference(FLAGS)
    print(metric)

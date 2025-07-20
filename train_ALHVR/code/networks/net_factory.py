from .unet import UNet_fea_aux
from .VNet import VNet_fea_aux
import re
def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    emb_num = re.findall('emb_(\d+)', net_type)
    if len(emb_num) > 0:
        emb_num = int(emb_num[0])
    else:
        emb_num = 0
    if net_type == "unet_fea_aux":
        net = UNet_fea_aux(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet_fea_aux" and mode == "train":
        net = VNet_fea_aux(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    return net
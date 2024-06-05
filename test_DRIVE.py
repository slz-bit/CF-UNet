import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
from model.unet import UNet
from model.unet_CTrans_Conv_LKA import UNet_CTrans_Conv_LKA
from test_dataset_DRIVE import test_dataset
from tester_DRIVE import Tester
from util.utils import BCELoss


def main(data_path, weight_path, CFG, show):
    checkpoint = torch.load(weight_path, map_location='cpu')
    # CFG_ck = checkpoint['config']
    Test_dataset = test_dataset(data_path, mode="test")
    test_loader = DataLoader(Test_dataset, batch_size=1,
                             shuffle=False,  num_workers=0, pin_memory=True)
    # model = get_instance(models, 'model', CFG)
    model = UNet_CTrans_Conv_LKA(in_chns=1, class_num=cfg['nclass'])
    loss =BCELoss()
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show)
    test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default=r"D:\datasets\DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default=r'', type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=True, action="store_true")
    args = parser.parse_args()
    with open(r"configs\acdc.yaml", encoding="utf-8") as file:
        cfg = Bunch(safe_load(file))
    main(args.dataset_path, args.wetght_path, cfg, args.show)

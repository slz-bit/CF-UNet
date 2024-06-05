import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
import util.utils
# from trainer import Trainer
from util.helpers import dir_exists, remove_files, double_threshold_iteration
from util.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
from PIL import Image

class Tester:
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=True):
        super(Tester, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = model.cuda()
        self.dataset_path = dataset_path
        self.show = show
        self.model.load_state_dict(checkpoint['model'])
        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }


    def test(self):

        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img, False)
                loss = self.loss(pre.softmax(dim=1).argmax(axis=1).unsqueeze(dim=1).float(), gt.float())
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300
                elif self.dataset_path.endswith("STARE"):
                    H, W = 605, 700



                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]

                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)

                    h, w = predict_b.shape
                    for m in range(0, h):
                        for n in range(0, w):
                            if predict_b[m][n] == 1:
                                predict_b[m][n] = 2
                            if predict_b[m][n] == 0:
                                predict_b[m][n] = 1
                            if predict_b[m][n] == 2:
                                predict_b[m][n] = 0

                    cv2.imwrite(
                        f"save_picture_48x48/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture_48x48/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture_48x48/pre{i}.png", np.uint8((1-predict)*255))
                    cv2.imwrite(
                        f"save_picture_48x48\pre_b{i}.png", np.uint8(predict_b*255))


                    # pre = torch.sigmoid(pre).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    # gt = torch.sigmoid(gt).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                self._metrics_update(
                    *get_metrics(pre, gt, self.CFG.threshold).values())

                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')

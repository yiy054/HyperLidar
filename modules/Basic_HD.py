from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modules.HDC_utils import set_model
from modules.ioueval import *
import torch.backends.cudnn as cudnn
from postproc.KNN import KNN
from common.avgmeter import *


from torchhd import functional
from torchhd import embeddings


VAL_CNT = 10

class BasicHD():
    def __init__(self, ARCH, DATA, datadir, modeldir, buffer_rate):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.modeldir = modeldir
        self.epochs = 20
        self.buffer_rate = buffer_rate

        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                        train_sequences=self.DATA["split"]["train"],
                                        valid_sequences=self.DATA["split"]["valid"],
                                        test_sequences=None,
                                        labels=self.DATA["labels"],
                                        color_map=self.DATA["color_map"],
                                        learning_map=self.DATA["learning_map"],
                                        learning_map_inv=self.DATA["learning_map_inv"],
                                        sensor=self.ARCH["dataset"]["sensor"],
                                        max_points=self.ARCH["dataset"]["max_points"],
                                        batch_size=self.ARCH["train"]["batch_size"],
                                        workers=self.ARCH["train"]["workers"],
                                        gt=True,
                                        shuffle_train=False)
        self.num_classes = self.parser.get_n_classes() 
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        # build model and criterion
        # self.model = model
        # concatenate the encoder and the head
        self.model = set_model(ARCH, modeldir, 'rp', 0, 0, self.num_classes, self.device)
        print(self.parser.get_n_classes())
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.parser.get_n_classes())
        print(self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def start(self):
        print("Starting training with the HDC online learning:")
        self.model.eval()
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        for e in range(1, 2):
            time1 = time.time()
            self.train(self.parser.get_train_set(), self.model)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(e, time2 - time1))
            # acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            # print('Stream final acc: {}'.format(acc))
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            time1 = time.time()
            self.retrain(self.parser.get_train_set(), self.model, epoch)

            time2 = time.time()
            print('retrain epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # final validation
            acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            print('Stream final acc: {}'.format(acc))

    def train(self, train_loader, model):  # task_list
        """Training on single-pass of data"""
        # Set validation frequency
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            self.mask = None
            train_time = []
            self.is_wrong_list = [None] * len(train_loader)  # store the wrong classification for each batch
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Training")):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                start = time.time()
                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(model.classify_weights.dtype)
                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1)
                proj_labels = proj_labels.to(self.device)
                
                
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                train_time.append(res)
                start = time.time()

                predictions =self.model.get_predictions(samples_hv)
                # print("predictions: ", predictions) #torch.Size([32768, 20])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                is_wrong = proj_labels != argmax

                # self.is_wrong_list[i] = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                # samples_hv = samples_hv[is_wrong]
                # samples_hv = samples_hv.to(model.classify_weights.dtype)
                # self.is_wrong_list[i] = is_wrong

                # # Pick the loss by the wrong
                # self.is_wrong_list[i] = proj_labels != argmax
                # loss = nn.CrossEntropyLoss(weight=self.loss_w.to(self.device))(predictions, proj_labels)
                # self.is_wrong_list[i] *= loss

                # # Save the loss by the Wy^X - WyX
                # # Compute dot products
                # true_scores = torch.sum(model.classify_weights[proj_labels] * samples_hv, dim=1)      # shape: [wrong_size]
                # wrong_scores = torch.sum(model.classify_weights[argmax] * samples_hv, dim=1) # shape: [wrong_size]
                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size]
                losses = wrong_scores - true_scores  # shape: [wrong_size]

                # # Initialize if needed — make sure it's a FloatTensor
                if self.is_wrong_list[i] is None or self.is_wrong_list[i].shape != is_wrong.shape:
                    self.is_wrong_list[i] = torch.zeros_like(is_wrong, dtype=losses.dtype)
                self.is_wrong_list[i][is_wrong] = losses

            model.classify.weight[:] = F.normalize(model.classify_weights)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))
            # print("Finish one batch, update classify weights")
    
    def retrain(self, train_loader, model, epoch):  # task_list
        """Training of one epoch on single-pass of data"""
        # Set validation frequency
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            total_miss = 0
            retrain_time = []
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Retraining")):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                start = time.time()
                model.classify.weight[:] = F.normalize(model.classify_weights)
                predictions, samples_hv, indices, self.is_wrong_list[i] = model(proj_in, self.mask, self.buffer_rate, self.is_wrong_list[i])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                # #proj_labels shape: torch.Size([1, 64, 512])
                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1) 
                proj_labels = proj_labels.to(self.device)
                proj_labels = proj_labels[indices]  # map to the sampled hypervectors

                is_wrong = proj_labels != argmax
                
                if is_wrong.sum().item() == 0:
                    continue

                # Check wrong classification number here and update the classify weights
                total_miss += is_wrong.sum().item()
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)
                # # Pick the loss by the wrong
                # true_scores = torch.sum(model.classify_weights[proj_labels] * samples_hv, dim=1)      # shape: [wrong_size]
                # wrong_scores = torch.sum(model.classify_weights[argmax] * samples_hv, dim=1) # shape: [wrong_size]
                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size
                losses = wrong_scores - true_scores  # shape: [wrong_size]

                # losses = true_scores - wrong_scores          # shape: [wrong_size]
                # print(losses.min(), losses.max())
                wrong_indices_within_selected = is_wrong.nonzero(as_tuple=False).squeeze()
                actual_wrong_indices = indices[wrong_indices_within_selected]
                self.is_wrong_list[i][actual_wrong_indices] = losses.to(self.is_wrong_list[i].dtype)

                # self.is_wrong_list[i][actual_wrong_indices] = True

                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                retrain_time.append(res)
                start = time.time()

            print("total_miss: ", total_miss)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC retraining time:{}\t std:{}".format(np.mean(retrain_time), np.std(retrain_time)))

    def validate(self, val_loader, model, evaluator):  # task_list
        """Validation, evaluate linear classification accuracy and kNN accuracy"""
        # return 0
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        evaluator.reset()
        validation_time = []
        class_func=self.parser.get_xentropy_class_string,
        with torch.no_grad():
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(val_loader, desc="Validation")):
                path_seq = path_seq[0]
                path_name = path_name[0]
                B, C, H, W = proj_in.shape[0], proj_in.shape[1], proj_in.shape[2], proj_in.shape[3]

                if self.gpu:
                    proj_in = proj_in.cuda()
                start = time.time()
                predictions, _, _, _ = model(proj_in, self.mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)
                start = time.time()

                predictions = predictions.view(B, H, W, self.num_classes)  # (1, H, W, C)
                predictions = predictions.permute(0, 3, 1, 2)        # → (1, C, H, W)
                argmax = predictions.argmax(dim=1)
                argmax = argmax.squeeze(0) 
                proj_labels = proj_labels.to(self.device)
                evaluator.addBatch(argmax, proj_labels)
            print("Mean HDC validation time:{}\t std:{}".format(np.mean(validation_time), np.std(validation_time)))
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), proj_in.size(0))
        iou.update(jaccard.item(), proj_in.size(0))

        print('Validation set:\n'
                'Time avg per batch xxx\n'
                'Loss avg {loss.avg:.4f}\n'
                'Jaccard avg {jac.avg:.4f}\n'
                'WCE avg {wces.avg:.4f}\n'
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(
                                                loss=losses,
                                                jac=jaccs,
                                                wces=wces,
                                                acc=acc, iou=iou))
        
        print('Class Jaccard: ', class_jaccard)
        return iou.avg
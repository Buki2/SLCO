import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm

from models.SLCO import mask_softmax
import json

class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            samples, targets = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.next_target = targets
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            self.next_target.tensor_dict = {k: tensor_dict[k].to(self.device, non_blocking=True) for k in tensor_dict}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target = self.next_img, self.next_mask, self.next_target
        self.preload()
        return img, mask, target

    def __next__(self):
        img, mask, target = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target

    def __iter__(self):
        return self

    def __len__(self):
        return self.length


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        knowledge_category_word_id, knowledge_category_word_mask = target_dict['knowledge_category_word_id'], target_dict['knowledge_category_word_mask']
        pseudo_annotation = target_dict['pseudo_annotation']
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs, subsentence_attn_list, pred_know_seg_score = model(img, mask, word_id, word_mask, knowledge_category_word_id, knowledge_category_word_mask)

        loss_dict = criterion(outputs, target_dict)
        weight_dict = criterion.weight_dict
        loss_dict['pseudo_loss'] = pseudo_strategy_loss(pred_know_seg_score, pseudo_annotation)
        loss_dict['div_loss'] = diversity_loss(subsentence_attn_list, word_mask)
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)

    output_know_seg = {}

    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        knowledge_category_word_id, knowledge_category_word_mask = target_dict['knowledge_category_word_id'], target_dict['knowledge_category_word_mask']
        pseudo_annotation = target_dict['pseudo_annotation']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs, subsentence_attn_list, pred_know_seg_score = model(img, mask, word_id, word_mask, knowledge_category_word_id, knowledge_category_word_mask)

        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict
            loss_dict['pseudo_loss'] = pseudo_strategy_loss(pred_know_seg_score, pseudo_annotation)
            loss_dict['div_loss'] = diversity_loss(subsentence_attn_list, word_mask)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

        sample_id = target_dict['sample_id'].cpu().numpy().tolist()
        word_id_list = word_id.cpu().numpy().tolist()
        fact_score_list = pred_know_seg_score.cpu().numpy().tolist()
        for i in range(len(sample_id)):
            output_know_seg[str(sample_id[i][0]) + '_' + str(sample_id[i][1])] = [word_id_list[i], fact_score_list[i]]

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}

    with open('./knowledge_retrieval/output_knowledge_score.json', 'w') as f:
        json.dump(output_know_seg, f)

    return val_stats, val_acc, val_time


def pseudo_strategy_loss(pred, target):
    target = torch.where(target < 0, torch.zeros_like(target), target)
    mseloss = torch.nn.MSELoss(reduction='mean')
    loss = mseloss(pred, target)
    return loss


def diversity_loss(score_list, word_mask, m=-1, coverage_reg=True):
    score_matrix = torch.stack([mask_softmax(score, word_mask) for score in score_list], dim=1)
    cov_matrix = torch.bmm(score_matrix, score_matrix.permute(0, 2, 1))
    id_matrix = torch.eye(cov_matrix.shape[1]).unsqueeze(0).repeat(cov_matrix.shape[0], 1, 1).cuda()
    if m == -1.:
        div_reg = torch.sum(((cov_matrix*(1-id_matrix))**2).view(-1))/cov_matrix.shape[0]
    else:
        div_reg = torch.sum(((cov_matrix-m*id_matrix)**2).view(-1))/cov_matrix.shape[0]
    if coverage_reg:
        word_mask_cp = word_mask.clone()
        for ii in range(word_mask_cp.shape[0]):
            word_mask_cp[ii, 0] = 0
            word_mask_cp[ii, word_mask_cp[ii, :].sum()] = 0
        cover_matrix = 1.-torch.clamp(torch.sum(score_matrix, dim=1, keepdim=False), min=0., max=1.)
        cover_reg = torch.sum((cover_matrix*word_mask_cp.float()).view(-1))/cov_matrix.shape[0]
        div_reg += cover_reg
    return div_reg

#!/usr/bin/env python
import sys
sys.path.append("/data/user1/code/UniDSeg")

import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from xmuda.common.solver.build import build_optimizer, build_optimizer_2d, build_optimizer_3d, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.losses import entropy_loss

from xmuda.data.build import build_dataloader  # contain depth and intensity info
from xmuda.models.UniDSeg_clip.build import build_model_2d, build_model_3d
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters 2D: {:.2e}'.format(num_params))

    num_params = sum(param.numel() for param in model_2d.net_2d.image_backbone.parameters())
    print('#Parameters: {:.2e}'.format(num_params))
    num_params_reins = []
    num_params_pg = []
    num_params_others = []
    for name, param in model_2d.net_2d.image_backbone.named_parameters():
        if 'reins' in name:
            num_params_reins.append(param.numel())
        elif 'prompt_generator' in name:
            num_params_pg.append(param.numel())
        else:
            num_params_others.append(param.numel())
    print('#Parameters reins: {:.2e}\n'.format(sum(num_params_reins)),
          '#Parameters pg: {:.2e}\n'.format(sum(num_params_pg)),
          '#Parameters others: {:.2e}'.format(sum(num_params_others)))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters 3D: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
    # optimizer_2d = build_optimizer(cfg, model_2d)  # for a2d2
    optimizer_2d = build_optimizer_2d(cfg, model_2d)  # 0.1x
    optimizer_3d = build_optimizer_3d(cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     # optimizer=optimizer_2d,
                                     # scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    # checkpoint_data_2d = checkpointer_2d.load(
    #     '/data/wuyao/code/UniDSeg/output/usa_singapore/UniDSeg_ViT-L-14/model_2d_060000.pth', resume=False)
    # checkpoint_data_2d = checkpointer_2d.load(
    #     '/data/wuyao/code/UniDSeg/output/day_night/UniDSeg_ViT-L-14/model_2d_100000.pth', resume=False)
    checkpoint_data_2d = checkpointer_2d.load(
        '/data1/wuyao/code/UniDSeg/output/a2d2_skitti/UniDSeg_ViT-L-14/model_2d_150000.pth', resume=False)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     # optimizer=optimizer_3d,
                                     # scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    # checkpoint_data_3d = checkpointer_3d.load(
    #     '/data/wuyao/code/UniDSeg/output/usa_singapore/UniDSeg_ViT-L-14/model_3d_060000.pth', resume=False)
    # checkpoint_data_3d = checkpointer_3d.load(
    #     '/data/wuyao/code/UniDSeg/output/day_night/UniDSeg_ViT-L-14/model_3d_100000.pth', resume=False)
    checkpoint_data_3d = checkpointer_3d.load(
        '/data1/wuyao/code/UniDSeg/output/a2d2_skitti/UniDSeg_ViT-L-14/model_3d_150000.pth', resume=False)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    # start_iteration = checkpoint_data_2d.get('iteration', 0)
    start_iteration = 0

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    # train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    if cfg.TRAIN.CLASS_WEIGHTS_PL:
        class_weights_pl = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS_PL).cuda()
    else:
        class_weights_pl = None

    # 冻结VFM，微调可学习层
    for name, param in model_2d.net_2d.image_backbone.named_parameters():
        if 'reins' in name or 'prompt_generator' in name:
            param.requires_grad = True
            param.lr = 0.1 * cfg.OPTIMIZER.BASE_LR
        else:
            param.requires_grad = False
        # print(name, param.requires_grad)

    setup_train()

    end = time.time()
    # train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        # _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            # data_batch_src['x'][0] = data_batch_src['x'][0].cuda()
            # data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            # data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            # data_batch_src['img'] = data_batch_src['img'].cuda()
            # data_batch_src['depth'] = data_batch_src['depth'].cuda()
            # target
            data_batch_trg['x'][0] = data_batch_trg['x'][0].cuda()
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            data_batch_trg['depth'] = data_batch_trg['depth'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
                data_batch_trg['pseudo_label_ensemble'] = data_batch_trg['pseudo_label_ensemble'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        # preds_2d = model_2d(data_batch_src)
        # preds_3d = model_3d(data_batch_src)
        #
        # # segmentation loss: cross entropy
        # seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        # seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        # train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d)
        # loss_2d = seg_loss_src_2d
        # loss_3d = seg_loss_src_3d
        #
        # if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
        #     # cross-modal loss: KL divergence
        #     seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
        #     seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
        #     xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
        #                               F.softmax(preds_3d['seg_logit'].detach(), dim=1),
        #                               reduction='none').sum(1).mean()
        #     xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
        #                               F.softmax(preds_2d['seg_logit'].detach(), dim=1),
        #                               reduction='none').sum(1).mean()
        #     train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
        #                                xm_loss_src_3d=xm_loss_src_3d)
        #     loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
        #     loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
        #
        # # update metric (e.g. IoU)
        # with torch.no_grad():
        #     train_metric_2d.update_dict(preds_2d, data_batch_src)
        #     train_metric_3d.update_dict(preds_3d, data_batch_src)
        #
        # # backward
        # loss_2d.backward()
        # loss_3d.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)

        loss_2d = []
        loss_3d = []
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # uni-modal self-training loss with pseudo labels
            # pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_trg['pseudo_label_2d'])
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_trg['pseudo_label_ensemble'],
                                             weight=class_weights_pl)
            # pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_trg['pseudo_label_3d'])
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_trg['pseudo_label_ensemble'],
                                             weight=class_weights_pl)
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)

        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_trg)
            train_metric_3d.update_dict(preds_3d, data_batch_trg)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr_2d: {lr_2d:.2e}',
                        'lr_3d: {lr_3d:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr_2d=optimizer_2d.param_groups[0]['lr'],
                    lr_3d=optimizer_3d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()

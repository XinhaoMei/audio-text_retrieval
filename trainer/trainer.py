#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent, WeightTriplet
from models.ASE_model import ASE
from data_handling.DataLoader import get_dataloader


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name

    folder_name = '{}_data_{}_freeze_{}_lr_{}_' \
                  'margin_{}_seed_{}'.format(exp_name, config.dataset,
                                             str(config.training.freeze),
                                             config.training.lr,
                                             config.training.margin,
                                             config.training.seed)

    log_output_dir = Path('outputs', folder_name, 'logging')
    model_output_dir = Path('outputs', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    model = ASE(config)
    model = model.to(device)

    # set up optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    elif config.training.loss == 'ntxent':
        criterion = NTXent()
    elif config.training.loss == 'weight':
        criterion = WeightTriplet(margin=config.training.margin)
    else:
        criterion = BiDirectionalRankingLoss(margin=config.training.margin)

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']

    # training loop
    recall_sum = []

    for epoch in range(ep, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

            audios, captions, audio_ids, _ = batch_data

            # move data to GPU
            audios = audios.to(device)
            audio_ids = audio_ids.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            loss = criterion(audio_embeds, caption_embeds, audio_ids)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)

        elapsed_time = time.time() - start_time

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {scheduler.get_last_lr()[0]:.6f}.')

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        r1, r5, r10, r50, medr, meanr = validate(val_loader, model, device)
        r_sum = r1 + r5 + r10
        recall_sum.append(r_sum)

        writer.add_scalar('val/r@1', r1, epoch)
        writer.add_scalar('val/r@5', r5, epoch)
        writer.add_scalar('val/r@10', r10, epoch)
        writer.add_scalar('val/r@50', r50, epoch)
        writer.add_scalar('val/med@r', medr, epoch)
        writer.add_scalar('val/mean@r', meanr, epoch)

        # save model
        if r_sum >= max(recall_sum):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'epoch': epoch,
            }, str(model_output_dir) + '/best_model.pth')

        scheduler.step()

    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    validate(test_loader, model, device)
    main_logger.info('Evaluation done.')
    writer.close()


def validate(data_loader, model, device):

    val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            # move data to GPU
            audios = audios.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate text to audio retrieval
        r1, r5, r10, r50, medr, meanr = t2a(audio_embs, cap_embs)

        val_logger.info('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1, r5, r10, r50, medr, meanr))

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(audio_embs, cap_embs)

        val_logger.info('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a))

        return r1, r5, r10, r50, medr, meanr


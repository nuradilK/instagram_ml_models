import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score

from utils import *
from model import MobileNetClf

global_step = 0


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, iterator, optimizer, criterion, scheduler, device, writer):
    epoch_loss = 0
    preds, targets = [], []
    global global_step
    
    model.train()
    pbar = tqdm(iterator, desc='Train epoch')
    for (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        loss = criterion(y_pred, y.float())
        
        preds += y_pred.tolist()
        targets += y.tolist()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        roc_auc = roc_auc_score(y.tolist() + [1], y_pred.tolist() + [1])
        pbar.set_description('Batch loss: {}, Batch ROC AUC: {}'.format(loss.item(), roc_auc))
        writer.add_scalar('Batch_Loss/train', loss.item(), global_step)
        writer.add_scalar('Batch_Accuracy/ROC_AUC', roc_auc, global_step)
        global_step += 1
        
    epoch_loss /= len(iterator)
    epoch_roc_auc = roc_auc_score(targets, preds)

    return epoch_loss, epoch_roc_auc


def evaluate(model, iterator, device, criterion=None):
    epoch_loss = 0
    preds, targets = [], []
    
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc='Eval'):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            if criterion:
                loss = criterion(y_pred, y.float())
                epoch_loss += loss.item()

            preds += y_pred.tolist()
            targets += y.tolist()

    if criterion:
        epoch_loss /= len(iterator)
    epoch_roc_auc = roc_auc_score(targets, preds)
        
    if criterion is None:
        return epoch_roc_auc
    return epoch_loss, epoch_roc_auc


def train(args, model: nn.Module, train_iterator, valid_iterator, device):
    START_LR = 1e-7
    END_LR = 10
    NUM_ITER = 100

    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    criterion = nn.BCELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(model, train_iterator, END_LR, NUM_ITER)
    lr = lrs[np.argmin(losses)]
    plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)

    print('Optimal LR is {}'.format(lr))
    # optimizer = optim.Adam([
    #         {'params': model.base.parameters(), 'lr': lr * 1e-2},
    #         {'params': model.fc.parameters(), 'lr': lr}
    #     ], lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr * 1e-3)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = args.epoch_num * STEPS_PER_EPOCH

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = lr,
        total_steps = TOTAL_STEPS)

    best_valid_roc_auc = float('inf')
    writer = SummaryWriter('tensorboard')

    try:
        for epoch in tqdm(range(args.epoch_num), desc='Training'):
            train_loss, train_roc_auc = train_epoch(model, train_iterator, optimizer, criterion, scheduler, device, writer)
            valid_loss, valid_roc_auc = evaluate(model, valid_iterator, device, criterion)

            if valid_roc_auc > best_valid_roc_auc:
                best_valid_roc_auc = valid_roc_auc
                torch.save(model.state_dict(), 'best_model.pt')

            print(f'\tTrain Loss: {train_loss:.3f} | Train ROC AUC: {train_roc_auc:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid ROC AUC: {valid_roc_auc:6.2f}%')

            writer.add_scalar('Epoch_Loss/train', train_loss, epoch)
            writer.add_scalar('Epoch_Loss/valid', valid_loss, epoch)
            writer.add_scalar('Epoch_ROC_AUC/train', train_roc_auc, epoch)
            writer.add_scalar('Epoch_ROC_AUC/valid', valid_roc_auc, epoch)
    except KeyboardInterrupt:
        print('Exiting the training')

    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-dir', required=True, help='Dataset directory path')
    parser.add_argument('--epoch-num', default=10, type=int)
    parser.add_argument('--train-ratio', default=0.8, type=float)
    parser.add_argument('--valid-ratio', default=0.1, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train_dir = os.path.join(args.dataset_dir, 'training')
    test_dir = os.path.join(args.dataset_dir, 'testing')
    val_dir = os.path.join(args.dataset_dir, 'validation')
    # prepare_dataset_dir(args, train_dir, test_dir, validation)
    train_data, valid_data, test_data = get_data(args, train_dir, test_dir, val_dir)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    train_iterator = data.DataLoader(train_data, 
                                    shuffle=True, 
                                    batch_size=args.batch_size,
                                    collate_fn=collate_fn)

    valid_iterator = data.DataLoader(valid_data, 
                                    batch_size=args.batch_size,
                                    collate_fn=collate_fn)

    test_iterator = data.DataLoader(test_data, 
                                    batch_size=args.batch_size,
                                    collate_fn=collate_fn,
                                    drop_last=True)

    model = MobileNetClf()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train(args, model, train_iterator, valid_iterator=valid_iterator, device=device)
    model.load_state_dict(torch.load('best_model.pt'))

    test_acc = evaluate(model, test_iterator, device=device)

    print('Test ROC AUC is {}'.format(test_acc))


if __name__ == '__main__':
    main()
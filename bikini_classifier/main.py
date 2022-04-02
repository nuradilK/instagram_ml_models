import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from sklearn.metrics import roc_auc_score

from utils import *
from model import MobileNetClf


def train_epoch(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_roc_auc = 0
    
    model.train()
    for (x, y) in tqdm(iterator, desc='Train epoch'):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        loss = criterion(y_pred, y.float())
        
        roc_auc = roc_auc_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_roc_auc += roc_auc.item()
        
    epoch_loss /= len(iterator)
    epoch_roc_auc /= len(iterator)
        
    return epoch_loss, epoch_roc_auc


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_roc_auc = 0
    
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc='Eval'):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y.float())

            roc_auc = roc_auc_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            epoch_loss += loss.item()
            epoch_roc_auc += roc_auc.item()
        
    epoch_loss /= len(iterator)
    epoch_roc_auc /= len(iterator)
        
    return epoch_loss, epoch_roc_auc


def train(args, model: nn.Module, train_iterator, valid_iterator):
    START_LR = 1e-7
    END_LR = 10
    NUM_ITER = 100

    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(model, train_iterator, END_LR, NUM_ITER)
    lr = lrs[np.argmin(losses)]
    plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = args.epoch_num * STEPS_PER_EPOCH

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = lr,
                                        total_steps = TOTAL_STEPS)

    best_valid_loss = float('inf')

    for epoch in tqdm(range(args.epoch_num), desc='Training'):
        start_time = time.monotonic()
        
        train_loss, train_roc_auc = train_epoch(model, train_iterator, optimizer, criterion, scheduler, device)
        valid_loss, valid_roc_auc = evaluate(model, valid_iterator, criterion, device)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train ROC AUC: {train_roc_auc:6.2f}% | ')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid ROC AUC: {valid_roc_auc:6.2f}% | ')


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset-dir', required=True, help='Dataset directory path')
    parser.add_argument('--epoch-num', default=10)
    parser.add_argument('--train-ratio', default=0.8)
    parser.add_argument('--valid-ratio', default=0.1)
    parser.add_argument('--batch-size', default=64)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_dir = os.path.join(args.dataset_dir, 'train')
    test_dir = os.path.join(args.dataset_dir, 'test')
    prepare_dataset_dir(args, train_dir, test_dir)
    train_data, valid_data, test_data = get_data(args, train_dir, test_dir)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    train_iterator = data.DataLoader(train_data, 
                                    shuffle = True, 
                                    batch_size = args.batch_size)

    valid_iterator = data.DataLoader(valid_data, 
                                    batch_size = args.batch_size)

    test_iterator = data.DataLoader(test_data, 
                                    batch_size = args.batch_size)

    model = MobileNetClf()
    train(args, model, train_iterator, valid_iterator=valid_iterator)


if __name__ == '__main__':
    main()
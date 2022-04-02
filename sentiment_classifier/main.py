import time
import sys
import copy
import torch 
import numpy as np
from scipy.sparse import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pyarrow as pa

import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertConfig, DistilBertTokenizer
from model import DistilBertForSequenceClassification

import pandas as pd

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class text_dataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __getitem__(self, index):
        tokenized_comment = TOKENIZER.tokenize(self.x[index])
        
        if len(tokenized_comment) > MAX_SEQ_LENGTH:
            tokenized_comment = tokenized_comment[:MAX_SEQ_LENGTH]
            
        ids_review  = TOKENIZER.convert_tokens_to_ids(tokenized_comment)
        padding = [0] * (MAX_SEQ_LENGTH - len(ids_review))
        ids_review += padding
        
        assert len(ids_review) == MAX_SEQ_LENGTH
        
        ids_review = torch.tensor(ids_review)
        if self.y is None:
            return ids_review 

        hcc = self.y[index] # toxic comment        
        list_of_labels = [torch.from_numpy(hcc)]

        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x)


def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    y_pred = y_pred.sigmoid() if sigmoid else y_pred
    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()


def preds(model, test_loader):
    predictions = []
    for inputs, sentiment in test_loader:
        inputs = inputs.to(DEVICE) 
        sentiment = sentiment.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            predictions.append(outputs.cpu().detach().numpy().tolist())
    return predictions


def train_model(model, criterion, optimizer, scheduler, dataloaders_dict, dataset_sizes, num_epochs=2):
    model.train()
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            micro_roc_auc_acc = 0.0

            # Iterate over data.
            for inputs, hcc in dataloaders_dict[phase]:
                inputs = inputs.to(DEVICE) 
                hcc = hcc.to(DEVICE)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs,hcc.float())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                micro_roc_auc_acc +=  accuracy_thresh(outputs.view(-1,6),hcc.view(-1,6))
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_micro_roc_acc = micro_roc_auc_acc / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} micro_roc_auc_acc: {:.4f}'.format( phase, epoch_micro_roc_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                    'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'distilbert_model_weights.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    ## Feature engineering to prepare inputs for BERT....
    Y = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].astype(float)
    X = train['comment_text']

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    config = DistilBertConfig(
        vocab_size=32000, hidden_dim=768,
        dropout=0.1, num_labels=6,
        n_layers=12, n_heads=12, 
        intermediate_size=3072)

    training_dataset = text_dataset(X_train,y_train)
    print('train dataset')
    test_dataset = text_dataset(X_test,y_test)
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=False),
        'val':torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }
    dataset_sizes = {
        'train':len(X_train),
        'val':len(X_test)
    }

    model = DistilBertForSequenceClassification(config)
    model.to(DEVICE)

    lrlast = .001
    lrmain = 3e-5
    #optim1 = torch.optim.Adam(
    #    [
    #        {"params":model.parameters,"lr": lrmain},
    #        {"params":model.classifier.parameters(), "lr": lrlast},
    #       
    #   ])

    optim1 = torch.optim.Adam(model.parameters(),lrmain)

    optimizer_ft = optim1
    criterion = nn.BCEWithLogitsLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft1 = train_model(
        model, criterion, optimizer_ft,
        exp_lr_scheduler, dataloaders_dict,
        dataset_sizes, num_epochs=8)

    #y_test = test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
    x_test = test['comment_text']
    y_test = np.zeros(x_test.shape[0]*6).reshape(x_test.shape[0],6)

    test_dataset = text_dataset(x_test,y_test)
    prediction_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = preds(model=model_ft1,test_loader=prediction_dataloader)
    predictions = np.array(predictions)[:,0]

    submission = pd.DataFrame(predictions,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
    test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']] = submission
    final_sub = test[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]

    final_sub.to_csv('submissions.csv', index=False)


if __name__ == '__main__':
    main()

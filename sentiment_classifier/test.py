import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from model import DistilBertForSequenceClassification
from transformers import DistilBertConfig, DistilBertTokenizer

TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 256


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


def preds(model, test_loader):
    predictions = []
    for inputs in tqdm(test_loader, total=len(test_loader)):
        inputs = inputs.to(DEVICE) 
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            predictions.append(outputs.cpu().detach().numpy().tolist())
    return predictions


def main():
    config = DistilBertConfig(
        vocab_size=32000, hidden_dim=768,
        dropout=0.1, num_labels=6,
        n_layers=12, n_heads=12, 
        intermediate_size=3072)
    model = DistilBertForSequenceClassification(config)
    model.load_state_dict(torch.load('distilbert_model_weights.pth'))
    model.to(DEVICE)
    # comments = [
    #     'Fuck u', 'I hate u', 'I wish u were dead!', 
    #     'U r a good man', 'I hope that you will die someday', 'Hate must be vanished',
    #     'Fuck yeah', 'Fucking aweasome', 'The government should have done somethinng more meaningful']

    # test_dataset = text_dataset(comments)
    # prediction_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # predictions = preds(model=model, test_loader=prediction_dataloader)
    # predictions = np.array(predictions)[:,0]

    test = pd.read_csv('./data/test.csv')
    x_test = test['comment_text']

    test_dataset = text_dataset(x_test)
    prediction_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = preds(model=model, test_loader=prediction_dataloader)
    predictions = np.array(predictions)[:,0]

    submission = pd.DataFrame(predictions,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
    test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]=submission
    final_sub = test[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    final_sub.to_csv('kaggle_submissions.csv',index=False)#


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse, os

from rouge import Rouge

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


rouge = Rouge()
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }



def train(epoch, tokenizer, model, device, loader, optimizer, args):
    model.train()
    for batch_idx, data in enumerate(loader, 0):
        #y = data['target_ids'].to(device, dtype = torch.long)
        #y_ids = y[:, :-1].contiguous()
        #lm_labels = y[:, 1:].clone().detach()
        #lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        lm_labels = data['target_ids'].to(device, dtype = torch.long)
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        #outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)

        loss = outputs[0]
        
        if batch_idx%args.print_every == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss:  {loss.item()}')
        
        loss = loss / args.gradient_accumulation
        loss.backward()
        
        if ((batch_idx + 1) % args.gradient_accumulation == 0) or (batch_idx + 1 == len(loader)):
            optimizer.step()
            optimizer.zero_grad()

def validate(epoch, tokenizer, model, device, loader, args):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=args.tgt_len, 
                num_beams=args.beam,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--ctx_len', type=int)
    parser.add_argument('--tgt_len', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--gradient_accumulation', type=int)
    parser.add_argument('--beam', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--gpuid', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_dir', type=str)
    return parser.parse_args()

def main(args):


    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    df = pd.read_csv(args.train_data,encoding='latin-1')
    df = df[['text','ctext']]
    df.ctext = 'summarize: ' + df.ctext
    print(df.head())

    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state = args.seed)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, args.ctx_len, args.tgt_len)
    val_set = CustomDataset(val_dataset, tokenizer, args.ctx_len, args.tgt_len)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size':  args.batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)

    print('Initiating Fine-Tuning for the model on our dataset')
    max_rouge = -1
    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, device, training_loader, optimizer, args)
        
        
        print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
        predictions, actuals = validate(1, tokenizer, model, device, val_loader, args)
        scores = rouge.get_scores(predictions, actuals, avg=True)    
        print(scores)
        rouge_f = round(scores['rouge-1']['f'], 6)
        if rouge_f > max_rouge:
            max_rouge = rouge_f
            final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
            final_df.to_csv(args.save_dir + '/e_'+str(epoch) + '_' + str(max_rouge) + '_predictions.csv')
            print('Output Files generated for review')


            print('save model')
            save_path = os.path.join(args.save_dir, 'epoch' + str(epoch) + '_' + "model_files")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    args = parse_config()
    main(args)
    exit(0)


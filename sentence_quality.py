import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from kobert_tokenizer import KoBertTokenizer


parser = argparse.ArgumentParser(description='BGM Recommendation given Text')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--ckpt_path',
                        type=str,
                        default='jhgan/ko-sbert-multitask',
                        help='need existed pretrained model ckpt')

        parser.add_argument('--sentence1_dataset',
                        type=str,
                        default="sent1.csv",
                        help='need sentence1 dataset')

        parser.add_argument('--sentence2_dataset',
                        type=str,
                        default="sent2.csv",
                        help='need sentence2 dataset')
        return parser

class contentDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = pd.read_csv(file)
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
        self.column = self.content.columns[0]
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]
        text = instance[self.column]
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_)}        

    def __len__(self):
        return self.len


def align_loss(x, y, alpha=2):    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def main(ckpt_path, sentence1_dataset, sentence2_dataset):

    def load_model(path):
        if 'kobert' in path:
            tok = KoBertTokenizer.from_pretrained(path)
        else:
            tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        return tok, model

    
    tok, model = load_model(ckpt_path)


    data_setup_1 = contentDataset(file = sentence1_dataset, tok = tok, max_len = 128)
    data_setup_2 = contentDataset(file = sentence2_dataset, tok = tok, max_len = 128)

    dataloader_1 = DataLoader(data_setup_1, batch_size=2, shuffle=False)
    dataloader_2 = DataLoader(data_setup_2, batch_size=2, shuffle=False)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    align_all = []
    unif_all = []

    for batch1, batch2 in zip(dataloader_1, dataloader_2):
        batch1 = {k: v.to(device) for k, v in batch1.items()}
        batch2 = {k: v.to(device) for k, v in batch2.items()}

        encoder_attention_mask1 = batch1["encoder_input_ids"].ne(0).float().to(device)
        encoder_attention_mask2 = batch2["encoder_input_ids"].ne(0).float().to(device)

        with torch.no_grad():
            outputs1 = model(batch1['encoder_input_ids'], attention_mask=encoder_attention_mask1)
            outputs2 = model(batch2['encoder_input_ids'], attention_mask=encoder_attention_mask2)

            if 'ELECTRA' in ckpt_path:
                pooler_output1 = outputs1.last_hidden_state
                pooler_output2 = outputs2.last_hidden_state
            else:
                pooler_output1 = outputs1.pooler_output
                pooler_output2 = outputs2.pooler_output

            pooler_output1 = F.normalize(pooler_output1,p=2,dim=1)
            pooler_output2 = F.normalize(pooler_output2,p=2,dim=1)

            align_all.append(align_loss(pooler_output1, pooler_output2, alpha=2))

             # print(align_all)
            
            if 'ELECTRA' in ckpt_path:

                
                pooler_output1 = pooler_output1[:,0,:]
                pooler_output1 = pooler_output1.squeeze(1)

                pooler_output2 = pooler_output2[:,0,:]
                pooler_output2 = pooler_output2.squeeze(1)

                pooler_cat = torch.cat((pooler_output1, pooler_output2))
                unif_all.append(uniform_loss(pooler_cat, t=2))
                
            else:

                pooler_cat = torch.cat((pooler_output1, pooler_output2))
                unif_all.append(uniform_loss(pooler_cat, t=2))
      

    alignment = sum(align_all) / len(align_all)

    uniformity = sum(unif_all) / len(unif_all)

    print('alignment: ',alignment)
    print('uniformity: ',uniformity)

if __name__ == '__main__':

    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()

    ko_sbert_multitask = 'jhgan/ko-sbert-multitask'
    msbert = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    kcelectra = "beomi/KcELECTRA-base"
    kosimcse = "BM-K/KoSimCSE-roberta"
    kobert = 'monologg/kobert'
    kodiffcse = "/home/keonwoo/anaconda3/envs/KoDiffCSE/sroberta_change_lr"
    koroberta = "klue/roberta-base"

    main(koroberta, args.sentence1_dataset, args.sentence2_dataset)
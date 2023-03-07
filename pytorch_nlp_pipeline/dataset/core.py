import torch
from torch.utils.data import Dataset, DataLoader
import logging

WORKER = '[bold]DATAMODULE[/bold]'

class TorchDataset(Dataset):
    """
    Build torch dataset
    
    Input:
        texts: 
        labels
        tokenizer
        max_len
        
    output:
        dictionary: used to construct torch dataloader
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
                                          text,
                                          add_special_tokens=True,
                                          truncation = True,
                                          max_length=self.max_len,
                                          return_token_type_ids=False,
                                          padding='max_length',
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                                )
        return {
                  'text': text,
                  'input_ids': encoding['input_ids'].flatten(),
                  'attention_mask': encoding['attention_mask'].flatten(),
                  'labels': torch.tensor(label, dtype=torch.float)
                }


class DatasetBase:
    def __init__(self, df, text_col, label_col, batch_size, max_len):
        logging.info(f'{WORKER}: Dataset initiating ...')
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.max_len = max_len

    def create_data_loader(self, tokenizer):
        """
        Construct data loader, on a torch dataset
        
        input:
            dataframe
            text column name
            label column name
            tokenizer
            max_len
            batch_size
            
        output:
            torch dataloader object
        """
        ds = TorchDataset(
                          texts=self.df[self.text_col].to_numpy(),
                          labels=self.df[self.label_col].to_numpy(),
                          tokenizer=tokenizer,
                          max_len=self.max_len
                          )
        return DataLoader(
                          ds,
                          batch_size=self.batch_size,
                          num_workers=0,
                          shuffle = False,
                          drop_last = False
                          )
    
    def data_preprocess(self):
        pass

    def get_data_stats(self):
        print(self.df.shape)
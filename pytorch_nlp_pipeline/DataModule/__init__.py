import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd



class build_torch_dataset(Dataset):
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


def split_data_w_sample(df, 
                        label_col, 
                        random_seed, 
                        stratify_col,
                        test_size = 0.1,
                        sample = None):
    """
    pass in original dataframe - train, val, test split - adjust sample numbers by tree in training data
    
    Input:
        dataframe
        label column name
        random seed
        test size 
        sample: dictionary default None - 
     
    
    output:
        df_train, df_test
        
    """
    
    stratify = df[stratify_col] if stratify_col else None

    
    # train val test split - stratified on label
    df_train, df_val = train_test_split(df,
                                          test_size=test_size,
                                          random_state=random_seed,
                                          stratify = stratify)

    
    
    if sample: # if choose number of samples by categories 
        df_list = []
        for label in sample:
            num_samples = sample[label]
            df_subset = df_train[df_train[label_col] == label]        
            if num_samples >df_subset.shape[0]:
                num_samples = df_subset.shape[0]
            
            df_list.append(df_subset.sample(num_samples))
       
        df_train = pd.concat(df_list)
      
    
    
    # data distribution check
    print('train: {}, val: {}'.format(df_train.shape[0], df_val.shape[0]))
    print('-' * 20)
    print('Lowest {} Counts'.format(label_col))
    train_min = df_train[label_col].value_counts().min()
    val_min = df_val[label_col].value_counts().min()
    print('train: {}. val: {}.'.format(train_min, val_min))
    
       
    return df_train.sample(frac=1, random_state = random_seed), df_val



class DataModule:

    def __init__(self, df, text_col, label_col, batch_size, max_len):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.max_len = max_len

    class build_torch_dataset(Dataset):
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


    def create_data_loader(self, df, tokenizer):
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
        ds = build_torch_dataset(
                                texts=df[self.text_col].to_numpy(),
                                labels=df[self.label_col].to_numpy(),
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
    


class ClfTrainDataset(DataModule):

    def __init__(self, 
                df,
                text_col, 
                label_col, 
                labels_to_indexes, 
                focused_indexes,
                batch_size,
                max_len, 
                random_seed,
                **kwargs):

        super().__init__(df, text_col, label_col, batch_size, max_len)
        
        self.random_seed = random_seed
        self.kwargs = kwargs
        
        self.labels_to_indexes = labels_to_indexes
        self.indexes_to_labels = {v: k for k, v in zip(labels_to_indexes.keys(), labels_to_indexes.values())}
        self.binary = False
        self.focused_indexes = focused_indexes


        if len(labels_to_indexes) == 1:
            self.binary = True



    def create_train_val(self, val_size): 
        df_train, df_val = split_data_w_sample(df=self.df, 
                                               label_col=self.label_col, 
                                               random_seed=self.random_seed, 
                                               stratify_col=self.label_col, 
                                               test_size=val_size,
                                               sample=self.kwargs.get('sample', None)
                                                                                )
        
        return df_train, df_val

        


        
        






   


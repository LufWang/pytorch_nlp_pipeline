from sklearn.model_selection import train_test_split
import logging
import pandas as pd

WORKER = 'DataModule'

def split_data_w_sample(df: pd.DataFrame, 
                        label_col: str, 
                        random_seed: int, 
                        stratify_col: str,
                        test_size = 0.1,
                        sample = None):
    """
    pass in original dataframe - train, test split - adjust sample numbers by tree in training data
    
    Input:
        dataframe
        label column name
        random seed
        stratify col
        test size 
        sample: dictionary default None - 
     
    
    output:
        df_train, df_test
        
    """
    logging.info(f'{WORKER}: Splitting dataset ...')
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
    logging.info(f'{WORKER}: train: {df_train.shape[0]}, val: {df_val.shape[0]}')
    logging.info(f'{WORKER}: Lowest {label_col} Counts')
    train_min = df_train[label_col].value_counts().min()
    val_min = df_val[label_col].value_counts().min()
    logging.info(f'{WORKER}: train: {train_min}. val: {val_min}.')
    
       
    return df_train.sample(frac=1, random_state = random_seed), df_val
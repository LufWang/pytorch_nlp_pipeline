import sys
sys.path.insert(0, '/Users/M255032/lufei-wang-local/piplelines/nlp_pipelines/pytorch_nlp_pipeline')
import pandas as pd
from pytorch_nlp_pipeline import model
from pytorch_nlp_pipeline.dataset import ClfDataset
from pytorch_nlp_pipeline.classification import Trainer
import random
from sklearn.model_selection import train_test_split
from torch import nn 


pretrained_path = '/Users/M255032/lufei-wang-local/pretrained_models/Bio_ClinicalBERT'



df = pd.read_csv('data/mri_train.csv')
text_col = 'note_text'
label_col = 'met_pos'
val_size = 0.2
batch_size = 6
max_len = 162
random_seed = 42

labels_to_indexes = {
    "met_pos": 0
}

focused_indexes = None

clf = model.PytorchNlpModel(
                        pretrained_type='BERT',
                        pretrained_path=pretrained_path,
                        device='cpu',
                        n_classes=len(labels_to_indexes),
                        freeze_pretrained=False,
                        head_hidden_size=512
                        )

df_train, df_val = train_test_split(df, test_size = 0.2)

train_data = ClfDataset(
                        df_train,
                        text_col,
                        label_col,
                        labels_to_indexes,
                        batch_size,
                        max_len,
                        random_seed=42

                    )

val_data = ClfDataset(
                        df_val,
                        text_col,
                        label_col,
                        labels_to_indexes,
                        batch_size,
                        max_len,
                        random_seed=42

                    )


print(f'df_train shape {df_train.shape} | df_val shape {df_val.shape}')

test_trainer = Trainer('cpu')

params = {
    "EPOCHS": 1,
    "lr": 0.00001,
    "weight_decay": 0.00001,
    "warmup_steps": 10
}

# test_trainer.train(clf, train_data, val_data, params)


### Test Multiclass
df['target'] = [random.randint(0, 2) for i in range(len(df))]

labels_to_indexes = {
    "0": 0,
    "1": 1,
    "2": 2
}

clf = model.PytorchNlpModel(
                        pretrained_type='BERT',
                        pretrained_path=pretrained_path,
                        device='cpu',
                        n_classes=len(labels_to_indexes),
                        freeze_pretrained=False,
                        head_hidden_size=512
                        )

print(type(nn.DataParallel(clf)) == nn.DataParallel)

df_train, df_val = train_test_split(df, test_size = 0.2, stratify = df.target)

train_data = ClfDataset(
                        df_train,
                        text_col,
                        'target',
                        labels_to_indexes,
                        batch_size,
                        max_len,
                        random_seed=42

                    )

val_data = ClfDataset(
                        df_val,
                        text_col,
                        'target',
                        labels_to_indexes,
                        batch_size,
                        max_len,
                        random_seed=42

                
                )



print(f'df_train shape {df_train.shape} | df_val shape {df_val.shape}')

params = {
    "EPOCHS": 1,
    "lr": 0.00001,
    "weight_decay": 0.00001,
    "warmup_steps": 10}




test_trainer.train(clf, train_data, val_data, params)

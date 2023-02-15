import sys
# sys.path.insert(0, '/Users/M255032/lufei-wang-local/piplelines/nlp_pipelines/pytorch_nlp_pipeline')
import pandas as pd
from pytorch_nlp_pipeline import model
from pytorch_nlp_pipeline.data import ClfTrainDataset
from pytorch_nlp_pipeline.text_clf import Trainer
import random



pretrained_path = '/Users/M255032/lufei-wang-local/pretrained_models/Bio_ClinicalBERT'
clf = model.BertModule(pretrained_path=pretrained_path)


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

data = ClfTrainDataset(
                        df,
                        text_col,
                        label_col,
                        labels_to_indexes,
                        None,
                        batch_size,
                        max_len,
                        random_seed=42

                    )

df_train, df_val = data.create_train_val(0.2)

print(f'df_train shape {df_train.shape} | df_val shape {df_val.shape}')

test_trainer = Trainer('cpu')

config = {
    "EPOCHS": 1,
    "lr": 0.00001,
    "weight_decay": 0.00001,
    "warmup_steps": 10
}

# test_trainer.train(clf, data, config, eval_freq=2)


### Test Multiclass
df['target'] = [random.randint(0, 2) for i in range(len(df))]

labels_to_indexes = {
    "0": 0,
    "1": 1,
    "2": 2
}
focused_indexes=[0, 1, 2]

data = ClfTrainDataset(
                        df,
                        text_col,
                        'target',
                        labels_to_indexes,
                        focused_indexes,
                        batch_size,
                        max_len,
                        random_seed=42

                    )

df_train, df_val = data.create_train_val(0.2)

print(f'df_train shape {df_train.shape} | df_val shape {df_val.shape}')

params = {
    "EPOCHS": 1,
    "lr": 0.00001,
    "weight_decay": 0.00001,
    "warmup_steps": 10}




test_trainer.train(clf, data, params)

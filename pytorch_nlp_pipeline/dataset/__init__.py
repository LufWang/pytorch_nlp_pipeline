from core import DatasetBase
import logging

WORKER = '[bold]DATAMODULE[/bold]'

class ClfDataset(DatasetBase):
    def __init__(self, 
                df,
                text_col, 
                label_col, 
                labels_to_indexes, 
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

        if len(labels_to_indexes) == 1:
            self.binary = True
            
        logging.info(f'{WORKER}: Classfication dataset initiated.')
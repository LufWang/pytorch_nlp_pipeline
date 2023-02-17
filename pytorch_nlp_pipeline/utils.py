import numpy as np
from google.cloud import storage
import os


def get_config(params):
    # random pick a set of params
    config = {}
    
    for name in params:
        choice = np.random.choice(params[name])

        if type(choice) == np.int64:
            choice = int(choice)
        elif type(choice) == np.float64:
            choice= float(choice)
        elif type(choice) == np.str_:
            choice = str(choice)

        config[name] = choice
    

    
    return config




class GCS_saver:

    def __init__(self, bucket_name):
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(bucket_name)

    
    def upload_from_memory(self, contents, destination_blob_name, content_type):
        blob = self.bucket.blob(destination_blob_name)

        blob.upload_from_string(contents, content_type=content_type)
    
    def upload_from_file(self, source_file_name, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

    def upload_torch_model(self, model_obj, destination_blob_name):
        import torch
        blob = self.bucket.blob(destination_blob_name)
        with blob.open("wb", ignore_flush=True) as f:
            torch.save(model_obj, f)

    def upload_pretrained_tokenizer(self, tokenizer, destination_blob_dir):

        # save tokenizer to local
        cwd = os.getcwd()
        saved_paths = tokenizer.save_pretrained(os.path.join(cwd, 'TEMP_FILES'))


        for path in saved_paths:

            fname = os.path.basename(path)
            blob_name = os.path.join(destination_blob_dir, fname)

            try:
                self.upload_from_file(path, blob_name)
            except:
                pass





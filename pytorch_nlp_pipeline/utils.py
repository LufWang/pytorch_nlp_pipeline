import numpy as np
# from google.cloud import storage
import os
import shortuuid
import json


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

def save_model(model, tokenizer, model_name, save_path, files):
        """
        model - model
        tokenizer - tokenizer intialized using transformer
        model_name - str: name of the saved directory
        save_path - path
        files - dict of files with key to be names and values to be files to be saved in the same directory, have to be all json files
        """
                        
        print('Saving Model...')
        
        # generate ID
        model_id = shortuuid.ShortUUID().random(length=12)
                        
        if not os.path.isdir(save_path):
            os.mkdir(save_path) # create directory if not exist

        # change here 
        save_path_final = os.path.join(save_path, model_id + '-' + model_name)

        if not os.path.isdir(save_path_final):
            os.mkdir(save_path_final) # create directory for model if not exist
        
        # save torch model
        torch.save(model.state_dict(), os.path.join(save_path_final, model_id + '-' + 'model.bin'))  # save model

        # save tokenizer
        tokenizer.save_pretrained(save_path_final)
        
        # save file in the files
        for file_name in files:
            with open(os.path.join(save_path_final, model_id + '-' + file_name), 'w', encoding='utf-8') as f:
                json.dump(files[file_name], f, ensure_ascii=False, indent=4)


        print('Model Files Saved.')
        print()


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




class GCS_reader:

    def __init__(self, bucket_name):
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(bucket_name)

    def read_json(self, blob_name):
        import json

        blob = self.bucket.blob(blob_name)

        with blob.open("r") as f:
            pass
            

from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch import nn, optim

from tqdm.auto import tqdm
import os 
from datetime import datetime
import json
import shortuuid
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score
from .utils import GCS_saver




class Trainer:

    def __init__(self, 
                 device
                 ):
    
    
        self.device = device


    def _get_loss_pred(self, outputs, labels, loss_fn, threshold, binary):
        """
        get loss and prediction from output of NN
        
        Args:
            outputs: output from model()
            binary: True or False
            loss_fn: loss function
            threshold: threshold to give a positive prediction (only for binary now)
        
        Returns:
            loss: pytorch loss
            preds: list of predicted labels
            
        """
        m = nn.Softmax(dim=1)
        
        if binary: # if doing binary classification
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, labels)
            preds_proba = np.array(torch.sigmoid(outputs).tolist()) # add sigmoid since no sigmoid in NN
            preds = np.where(preds_proba > threshold, 1, 0)
            
            return loss, preds, preds_proba, [1-preds_proba, preds_proba]
                
        else: # if doing multiclass 
            m = nn.Softmax(dim=1)
            loss = loss_fn(outputs, labels.long())
            # _, preds = torch.max(outputs, dim=1)
            preds_proba, preds = torch.max(m(outputs), dim=1)
        
        
            return loss, preds, preds_proba, m(outputs)

    def _eval_model(self, model, data_loader, loss_fn, device, threshold = 0.5, binary = True):

        model.eval()
        losses = []
        preds_l = []
        preds_probas_l = []
        true_labels_l = []
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)
                # labels = torch.nn.functional.one_hot(labels.to(torch.int64)).to(device).to(float) # one hot label to the right shape

                outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                                )

                
                loss, preds, preds_probas, preds_probas_all = self._get_loss_pred(outputs, labels, loss_fn, threshold, binary)
                


                preds_l.extend(preds.tolist())
                true_labels_l.extend(labels.tolist())
                preds_probas_l.extend(preds_probas.tolist())
                
                losses.append(loss.item())
                
        preds_l = np.array(preds_l)
        true_labels_l = np.array(true_labels_l)
        preds_probas_l = np.array(preds_probas_l)
        
        
        
        return preds_l, preds_probas_l, true_labels_l, losses

    def _evaluate_by_metrics(self, y_true, y_pred, metrics_list, average = 'binary', verbose = True):

        """
        Helper function that prints out and save a list of metrics defined by the user
        """

        results = {}
        output_str = ''

        for metric_name in metrics_list:
            metric_func = metrics_list[metric_name]

            if metric_name == 'confusion_matrix':
                score = metric_func(y_true, y_pred)
                score = score.tolist()
                output_str += f'| {metric_name}: {score} '

                results[metric_name] = score

            else:
                score = metric_func(y_true, y_pred, average=average, zero_division=0)
                results[metric_name] = score
                
                if type(score) == float:
                    score = round(score, 4)

                output_str += f'| {metric_name}: {score} '


        
        if verbose:
            print(output_str)

        return results

    # TODO change this to save into a GCS BUCKET
    def _save_model(self, model, tokenizer, model_name, save_path, files):
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
        save_path_final = os.path.join(save_path, model_id + '|' + model_name)

        if not os.path.isdir(save_path_final):
            os.mkdir(save_path_final) # create directory for model if not exist
        
        # save torch model
        torch.save(model.state_dict(), os.path.join(save_path_final, model_id + '|' + 'model.bin'))  # save model

        # save tokenizer
        tokenizer.save_pretrained(save_path_final)
        
        # save file in the files
        for file_name in files:
            with open(os.path.join(save_path_final, model_id + '|' + file_name), 'w', encoding='utf-8') as f:
                json.dump(files[file_name], f, ensure_ascii=False, indent=4)


        print('Model Files Saved.')
        print()

        return save_path_final


    def _save_model_GCS(self, model, tokenizer, model_name, files, bucket_name, dir_path):
        print('Saving Model...')

        saver = GCS_saver(bucket_name)
        
        # generate ID
        model_id = shortuuid.ShortUUID().random(length=12)

        # change here 
        blob_name_dir = os.path.join(dir_path, model_id + '-' + model_name)

        
        # save torch model
        model_name = model_id + '-' + 'model.bin'
        blob_name = os.path.join(blob_name_dir, model_name)
        saver.upload_torch_model(model, blob_name)

        # save tokenizer
        saver.upload_pretrained_tokenizer(tokenizer, blob_name_dir)
        
        # save file in the files
        for file_name in files:
            file = json.dumps(files[file_name], ensure_ascii=False, indent=4)
            file_name = model_id + '-' + file_name
            blob_name = os.path.join(blob_name_dir, file_name)
            saver.upload_from_memory(file, blob_name, content_type='application/json')


        print('Model Files Saved.')
        print()



    def train(self, 
              ModelModule,
              DataModule,
              params,
              eval_config = {
                "val_size": 0.2,
                "save_metric": f1_score,
                "threshold": 0,
                "gcs_bucket": None,
                "gcs_blob_dir": None,
                "save_model_name": " ",
                "eval_freq": 1,
                "watch_list": {
                    "F1": f1_score,
                    "Precision": precision_score,
                    "Recall": recall_score
                }
              },
              early_stopping = None,
                                ):
        ## set up data 
        binary = DataModule.binary
        labels_to_indexes = DataModule.labels_to_indexes
        indexes_to_labels = DataModule.indexes_to_labels
        focused_indexes = DataModule.focused_indexes
        label_col = DataModule.label_col
        threshold = 0
        df_train, df_val = DataModule.create_train_val(eval_config['val_size'])
        
        # unpack variables in config
        watch_list = eval_config['watch_list']
        gcs_bucket = eval_config['gcs_bucket']
        gcs_blob_dir = eval_config['gcs_blob_dir']
        save_path = eval_config['save_path']
        save_metric = eval_config['save_metric']
        eval_freq = eval_config['eval_freq']
        best_val_score = eval_config['threshold']


                # Checking for Binary of Multiclass
        if binary: # binary
            RATIO = df_train[df_train[label_col] == 0].shape[0] / df_train[df_train[label_col] == 1].shape[0]
            loss_fn = nn.BCEWithLogitsLoss(
                                        pos_weight = torch.tensor(RATIO )
                                        ).to(self.device)
            focused_indexes = None
            threshold = 0.5
            num_classes = 1
        
        else: # multiclassfication

            ## Assign class weights
            class_weight = [] 
            sample = df_train[label_col].value_counts().to_dict()

            for label in indexes_to_labels:
                class_weight.append(max(sample.values()) / sample[label])
            if focused_indexes: # if focused index boost their weights 
                for index in focused_indexes:
                    class_weight[index] = class_weight[index] * float(params.get('boost', 1))
        
            loss_fn = nn.CrossEntropyLoss(
                                                weight = torch.tensor(class_weight).to(self.device)
                                                ).to(self.device)

            num_classes = len(labels_to_indexes)

        

        # load tokenizer and model
        tokenizer = ModelModule.tokenizer 
        model = ModelModule.load_pretrained(num_classes, self.device)
        model = model.to(self.device)

        train_data_loader = DataModule.create_data_loader(df_train, tokenizer)
        val_data_loader = DataModule.create_data_loader(df_val, tokenizer)


        # get list eval steps based on eval_freq
        epoch_steps = len(train_data_loader)
        total_steps = epoch_steps * params['EPOCHS']
        total_evaluations = eval_freq * params['EPOCHS']
        print(f'Total Training Steps: {total_steps}')
        eval_steps = [int(total_steps/total_evaluations) * i for i in range(1, total_evaluations)]
        eval_steps.append(total_steps)
        print(f'Eval at steps: {eval_steps}')

        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay = params['weight_decay']) # optimizer to update weights
        scheduler = get_linear_schedule_with_warmup(
                                                    optimizer,
                                                    num_warmup_steps=params['warmup_steps'],
                                                    num_training_steps=total_steps
        )



        global_step = 1
        eval_ind = 0
        val_losses_list = [] # record val loss every eval step -> for early stopping
        running_train_loss = 0
        patience_count = 0
        val_scores_list = []
        best_model_path = None

         # start training
        EPOCHS = params['EPOCHS']
        for epoch in range(EPOCHS):

            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            print('Training...')

            losses = []
            train_preds_l = []
            train_true_labels_l = []
            
            # training through the train_data_loader
            for d in tqdm(train_data_loader):
                
                model.train()

                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device) 

                # getting output on current weights
                outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )


                # getting loss and preds for the current batch
                loss, preds, preds_proba, preds_proba_all = self._get_loss_pred(outputs, labels, loss_fn, threshold, binary)

                # backprogogate and update weights/biases
                losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


                # record performance
                running_train_loss += loss.item() # update running train loss
                train_preds_l.extend(preds.tolist())
                train_true_labels_l.extend(labels.tolist())

                # evaluating based on step
                if global_step == eval_steps[eval_ind]:
                    print()
                    eval_ind += 1
                    
                    print()
                    print(f'Evaluateing at Step {global_step}....')
                    val_preds, val_preds_probas, val_trues, val_losses = self._eval_model(
                                                                                            model,
                                                                                            val_data_loader,
                                                                                            loss_fn,
                                                                                            self.device,
                                                                                            threshold,
                                                                                            binary
                                                                                                    )
                    val_score_by_label = {}

                    if binary:
                        average = 'binary'
                    else:
                        average = 'macro'

                    
                    if focused_indexes: # if focused_indexes are passed in (multiclass only)

                        eval_results = self._evaluate_by_metrics(val_trues, val_preds, watch_list, average = None, verbose=False)
                        val_score_all = save_metric(val_trues, val_preds, average=None, zero_division=0)
                        
                        # print out scores to console
                        data_all = []
                        for index in focused_indexes:
                            data = []
                            label_name = indexes_to_labels[index]
                            data.append(label_name)
                            for metric_name in eval_results:
                                score = round(eval_results[metric_name][index], 3)
                                data.append(score)
                            
                            data_all.append(data)
                            val_score_by_label[indexes_to_labels[index]] = round(val_score_all[index], 3)
                        
                        print()
                        print(tabulate(data_all, headers=['Label'] + list(eval_results.keys())))
                        print()
                            
                            
                        val_score = np.mean(val_score_all[focused_indexes])
                    
                    else: # if not focused index or binary
                        eval_results = self._evaluate_by_metrics(val_trues, val_preds, watch_list, average = average, verbose=True)
                        val_score = save_metric(val_trues, val_preds, average = average)

                    print(f'Overall Score: {val_score}')
                    print()

                    val_scores_list.append(val_score)          
                    val_loss = np.mean(val_losses) # getting average val loss
                    val_losses_list.append(val_loss)

                    # check if needed to be early stopped: 
                    if early_stopping:
                        if patience_count > early_stopping:
                            if val_scores_list[-1] > val_scores_list[-(early_stopping + 1)]:
                                print('Early Stopping..')
                                print('Val F1 List: ', val_scores_list)
                                return None, None
                    
                        patience_count += 1

                    # if a save path provided, better models will be checkpointed
                    if val_score > best_val_score: # if f1 score better. save model checkpoint
                        
                        if save_path:
                            save_model_name = eval_config['save_model_name']

                            
                            model_info = {
                                    'val_score': val_score,
                                    'val_loss': float(np.round(np.mean(val_losses), 4)),
                                    'time_generated': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                                    'focused_labels':  [indexes_to_labels[x]for x in focused_indexes] if focused_indexes  else [],
                                    'val_score_by_focused_label': val_score_by_label
                                }
                    
                            files = {
                                'hyperparameters.json': params,
                                'model_info.json': model_info,
                                'labels_to_indexes.json': labels_to_indexes,
                                'indexes_to_labels.json': indexes_to_labels
                                
                            }
                            
                            # best_model_path = self._save_model_GCS(model, tokenizer, save_model_name, files, gcs_bucket, gcs_blob_dir)
                            best_model_path = self._save_model(self, model, tokenizer, save_model_name, save_path, files)

                        best_val_score = val_score # update best f1 score

                global_step += 1 # update training step count 

        return best_val_score, best_model_path
                    
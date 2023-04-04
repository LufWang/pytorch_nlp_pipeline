from transformers import BertModel, BertTokenizer, BioGptTokenizer, BioGptModel
import logging
from torch import nn
from collections import OrderedDict
from typing import Union



WORKER = 'ModelModule'



def construct_model_head(input_size: int, 
                         hidden_layers: Union[list, None], 
                         num_classes: int) -> nn.Sequential:
        """
        Construct classifier head layers 

        Input:
            input_size => int
            hidden_size => if list, layers will be added sequatially based on list order
            num_classes => int

        Returns:
            torch.nn.Sequential
        """

        seq_layers = []
        layer_num_counter = 1

    

        if hidden_layers:
            hidden_layers.append(num_classes)
        
        
        for layer in hidden_layers:
            if layer == 'relu':
                seq_layers.append((f'out{layer_num_counter}_relu', nn.ReLU()))
            else:
                seq_layers.append((f'out{layer_num_counter}_linear', nn.Linear(input_size, layer)))
                input_size = layer
        
            layer_num_counter+=1

        logging.info(f'{WORKER}: Classfication Head Added. Number of Hidden Layers - {len(seq_layers)}. Number of Classes - {num_classes}')

        
        return nn.Sequential(OrderedDict(seq_layers))



class SimpleNN(nn.module):
    def __init__(self, 
            input_size,
            output_size,
            hidden_layers: Union[list, None] =[384, 'relu']
            
                ):
        
        logging.info(f'{WORKER}: Simple Pytorch NN initiating...')

        super(SimpleNN, self).__init__()

        self.network = construct_model_head(input_size, hidden_layers, output_size)

        logging.info(f'{WORKER}: Simple Pytorch NN Initiated..')


    def forward(self, x):
        
        outputs = self.network(x)

        return outputs
        

    
 

class TransformerNN(nn.Module):
    def __init__(self, 
                 pretrained_type: str, 
                 pretrained_path: str, 
                 n_classes: int, 
                 freeze_pretrained: bool=True, 
                 head_hidden_layers: Union[list, None] =[384, 'relu']

                 ):
        """
        Pytorch nn.Module that also includes tokenizer and can load pretrained models - for text classification

        input: 
            pretrained_type => str: e.g. 'BERT'
            pertrained_path => str: local path to pretrained model dir or repo on hugging face hub
            n_classes => int: number of classes to classify
            freeze_pretrained => bool: whether to freeze pretrained layers
            head_hidden_layers => list or None: hidden layers to add to classfication head
        
        Returns:
            torch.nn.Module like pytorch model object
        
        """
        
        logging.info(f'{WORKER}: TransformerNN initiating...')
        super(TransformerNN, self).__init__()
        self.pretrained_type = pretrained_type

        if pretrained_type == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
            self.pretrained = BertModel.from_pretrained(pretrained_path)
        elif pretrained_type == 'BioGPT':
            self.tokenizer = BioGptTokenizer.from_pretrained(pretrained_path)
            self.pretrained = BioGptModel.from_pretrained(pretrained_path)

        logging.info(f'{WORKER}: tokenizer and pretrained for {pretrained_type} loaded.')
        self.pretrained_path = pretrained_path
        self.freeze_pretrained = freeze_pretrained
        self.drop = nn.Dropout(p=0.3)
        self.n_classes = n_classes
        print(head_hidden_layers)
        self.head = construct_model_head(self.pretrained.config.hidden_size,
                                      head_hidden_layers,
                                      self.n_classes)
        if freeze_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False
                
        logging.info(f'{WORKER}: TransformerNN of type {pretrained_type} with classification head initiated.')
        logging.info(f'{WORKER}: pretrained model freezed - {freeze_pretrained}')

    
    def forward(self, input_ids, attention_mask):
        outputs  = self.pretrained(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        if self.pretrained_type == 'BERT':
            outputs = self.drop(outputs[1])
            outputs = self.head(outputs)
        elif self.pretrained_type == 'BioGPT':
            outputs = self.head(outputs.last_hidden_state[:,0,:])
        return outputs
from transformers import BertModel, BertTokenizer, BioGptTokenizer, BioGptModel
import logging
from torch import nn

WORKER = '[bold]ModelModule[/bold]'

class ModelHead(nn.Module):
    def __init__(self, input_size=512, hidden_size=None, num_classes=1):
        super(ModelHead, self).__init__()
        self.hidden_layers = []
        if type(hidden_size) is not list:
            hidden_size = [hidden_size]
        hidden_size = [input_size] + hidden_size
        for layer_num in range(len(hidden_size)-1):
            self.hidden_layers.append(nn.Linear(hidden_size[layer_num], hidden_size[layer_num+1]))
        self.output_layer = nn.Linear(hidden_size[-1], num_classes)
        logging.info(f'{WORKER}: Classfication Head Added. Number of Hidden Layers - {len(self.hidden_layers)}. Number of Classes - {num_classes}')
    
    def forward(self, pretrained_output):
        x = pretrained_output
        for layer in self.hidden_layers:
            x  = layer(x)
        outputs = self.output_layer(x)
        return outputs


class PytorchNlpModel(nn.Module):
    def __init__(self, pretrained_type, pretrained_path, n_classes, freeze_pretrained=True, head_hidden_size=512):
        logging.info(f'{WORKER}: PytorchNlpModel initiating...')
        super(PytorchNlpModel, self).__init__()
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
        self.head = ModelHead(self.pretrained.config.hidden_size,
                                      head_hidden_size,
                                      self.n_classes)
        if freeze_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False
                
        logging.info(f'{WORKER}: PytorchNlpModel of type {pretrained_type} with classification head initiated.')
        logging.info(f'{WORKER}: pretrained model freezed - {freeze_pretrained}')

    def load_weights(self):
        pass
    
    def forward(self, input_ids, attention_mask):
        outputs  = self.pretrained(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        if self.pretrained_type == 'BERT':
            outputs = self.drop(outputs[1])
            outputs = self.head(outputs)
        elif self.pretrained_type == 'BioGPT':
            outputs = self.head(outputs.last_hidden_state[:,0,:])
        return outputs
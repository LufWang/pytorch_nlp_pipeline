from transformers import BertModel, BertTokenizer
from transformers import BioGptTokenizer, BioGptModel

from torch import nn



# Model Structure       
class CustomBertMultiClassifier(nn.Module):
    """
    Neural Network Structure
    
    """
    
    def __init__(self, pretrained_path, n_classes, device):
        super(CustomBertMultiClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes).to(device)
        self.n_classes = n_classes
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        outputs  = self.bert(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        outputs = self.drop(outputs[1]).to(self.device)
        
        
        outputs = self.out(outputs)
        
        return outputs




class CustomBioGPTMultiClassifier(nn.Module):
    """
    Neural Network Structure
    
    """
    
    def __init__(self, pretrained_path, n_classes, device):
        super(CustomBioGPTMultiClassifier, self).__init__()
        self.biogpt = BioGptModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.biogpt.config.hidden_size, n_classes).to(device)
        self.n_classes = n_classes
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        outputs  = self.biogpt(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        outputs = self.drop(outputs.last_hidden_state[:,0,:]).to(self.device)
        
        
        outputs = self.out(outputs)
        
        return outputs

class ModelModule:

    def __init__(self):
        pass


    def load_weights(self):
        pass


###
# TODO should be able to load different kinds of pretrained models
###
class BertModule:
    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)


    def load_pretrained(self, n_classes, device):
        return CustomBertMultiClassifier(self.pretrained_path, n_classes, device)



class BioGPTModule:
    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path
        self.tokenizer = BioGptTokenizer.from_pretrained(pretrained_path)



    def load_pretrained(self, n_classes, device):
        return CustomBioGPTMultiClassifier(self.pretrained_path, n_classes, device)



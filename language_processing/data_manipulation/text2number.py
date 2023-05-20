from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer, ElectraTokenizer
import numpy as np
import tensorflow as tf

class Text2Number():
    def __init__(self, batch_size=32):
        self.token = None
        self.batch_size = batch_size

    def tokenize_bert(self, data, seq_len=128, model="bert-base-cased"):
        '''
            Computationally expensive due to their large size, and the tokenization process can be slower
        '''
        tokenizer = BertTokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def tokenize_gpt2(self, data, seq_len=128, model="gpt2-medium"):
        '''
            Unidirectional
        '''
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def tokenize_robert(self, data, seq_len=128, model="roberta-base"):
        '''
            Requiring more computational resources for training and inference.
        '''
        tokenizer = RobertaTokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def tokenize_distilBERT(self, data, seq_len=128, model="distilbert-base-uncased"):
        '''
            Scenarios where memory and computational resources are limited
        '''
        tokenizer = DistilBertTokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def tokenize_Albert(self, data, seq_len=128, model="albert-base-v2"):
        '''
            Fewer parameters, making them more memory-efficient and faster during training and inference
        '''
        tokenizer = AlbertTokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def tokenize_Electra(self, data, seq_len=128, model="google/electra-base-discriminator"):
        '''
            Rely on a discriminator-generator setup, which means they may not capture certain aspects of language semantics
        '''
        tokenizer = ElectraTokenizer.from_pretrained(model)
        token = tokenizer.encode_plus(data.tolist(),
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)
        self.token = token

    def data_tokenized(self, data, labels, seq_len=128):
        '''
            Creates the train and validation datasets
        '''
        train_ds_ids = []
        train_ds_mask = []
        train_label = []
        labels = np.array(labels.tolist()).reshape(-1,1)

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for i, val in enumerate(data.tolist()):
            val = tokenizer.encode_plus(val,
                          max_length=seq_len,
                          truncation=True,
                          padding='max_length',
                          add_special_tokens=True,
                          return_attention_mask=True,)

            train_ds_ids.append(val['input_ids'])
            train_ds_mask.append(val['attention_mask'])
            train_label.append(labels[i])

        train_ds_ids = tf.convert_to_tensor(train_ds_ids)
        train_ds_mask = tf.convert_to_tensor(train_ds_mask)
        train_label = tf.convert_to_tensor(train_label)

        return train_ds_ids,train_ds_mask, train_label
    
    def map_func_test(self, input_ids, masks):
        return {'input_ids':input_ids,
                'attention_mask':masks}
    
    def map_func(self, input_ids, masks, labels):
        return {'input_ids':input_ids,
                'attention_mask':masks}, labels
    
    def get(self):
        return self.token
    
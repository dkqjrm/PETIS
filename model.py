import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
from matplotlib import pyplot as plt
import os
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel, AdamW, BertModel, BertConfig, RobertaConfig, RobertaModel
from dataset import CustomDataset
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torchmetrics
from prefix_encoder import PrefixEncoder


class Classifier(LightningModule):
    def __init__(self, config):  # drop_prob를 활용하여 dropout 추가해볼 것.
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)

        self.train_logit_list = []
        self.train_label_list = []
        self.valid_logit_list = []
        self.valid_label_list = []
        self.test_logit_list = []
        self.test_label_list = []

        self.robertaconfig = RobertaConfig.from_pretrained(
            self.config['model_name']
        )

        self.model = RobertaModel.from_pretrained(self.config['model_name'], config=self.robertaconfig)

        for param in self.model.parameters():
            param.requires_grad = False
        self.pre_seq_len = self.config['pre_seq_len']
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.n_embd = self.config['hidden_size'] // config['num_attention_heads']

        self.dropout = torch.nn.Dropout(config['hidden_dropout_prob'])
        self.f_mlp = nn.Sequential(
            nn.Linear(1024, len(self.tokenizer.vocab))
        )

    def forward(self, x):
        # model_input: [batch_size, 1952]
        # input_ids, attention_mask : [batch_size,  maxlen(512)]
        batch_size = x['input_ids'].shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.device)
        x['attention_mask'] = torch.cat((prefix_attention_mask, x['attention_mask']), dim=1)

        last_hidden_state = self.model(input_ids=x['input_ids'],
                                       attention_mask=x['attention_mask'],
                                       past_key_values=past_key_values).last_hidden_state  # [batch_size, seq_len, 1024]
        last_hidden_state = self.dropout(last_hidden_state)
        return self.f_mlp(last_hidden_state) # [1024, 32000]
        # [batch_size, seq_len, 32000]

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        sch = self.lr_schedulers()
        sch.step()

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)

        loss = self.criterion(out.transpose(1, 2), train_batch['label'])
        # out : batch, seq_len, vocab_size -> [batch, vocab_size, seq_len]
        # label : batch, seq_len
        output = self.train_metrics(torch.argmax(out, dim=2), train_batch['label'])

        # self.train_logit_list.append(out.detach())
        # self.train_label_list.append(train_batch['label'].detach())

        # sch = self.lr_schedulers()
        # sch.step()

        self.log_dict({'train/loss': loss.item(),
                       'train/acc': output['train_MulticlassAccuracy'],
                       'train/pre': output['train_MulticlassPrecision'],
                       'train/rec': output['train_MulticlassRecall'],
                       'train/f1': output['train_MulticlassF1Score'],
                       'lr': self.optimizers().param_groups[0]['lr']},
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def prepare_data(self):
        self.train_dataset = CustomDataset(self.config['path'], self.config['model_name'], self.config['pre_seq_len'])
        # self.val_dataset = StudDataset('../val14.tsv', self.config['prompt'] , self.config['pre_seq_len'])
        # self.test_dataset = StudDataset('../test14.tsv', self.config['prompt'] , self.config['pre_seq_len'])

        metrics = torchmetrics.MetricCollection([
            Accuracy(task='multiclass', num_classes=len(self.tokenizer.vocab), ignore_index=1, average='macro'),
            Recall(task='multiclass', num_classes=len(self.tokenizer.vocab), ignore_index=1, average='macro'),
            Precision(task='multiclass', num_classes=len(self.tokenizer.vocab), ignore_index=1, average='macro'),
            F1Score(task='multiclass', num_classes=len(self.tokenizer.vocab), ignore_index=1, average='macro')
        ])

        # metrics = torchmetrics.MetricCollection([
        #     Precision(task='binary', threshold=0.5),
        #     Recall(task='binary', threshold=0.5),
        #     F1Score(task='binary', threshold=0.5)
        # ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.train_epoch_metrics = metrics.clone(prefix='train_epoch_')
        self.valid_epoch_metrics = metrics.clone(prefix='val_epoch_')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"], collate_fn=self.train_dataset.custom_collate,
                          shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.config["batch_size"],
    #                       num_workers=self.config["num_workers"], collate_fn=self.val_dataset.custom_collate,
    #                       shuffle=False)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.config["batch_size"],
    #                       num_workers=self.config["num_workers"], collate_fn=self.test_dataset.custom_collate,
    #                       shuffle=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["learning_rate"],
            epochs=self.config['epoch'],
            steps_per_epoch=len(self.train_dataloader()) // self.config['accumulate'] + 1,
            anneal_strategy='linear',
            pct_start=0.1
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import random
import jsonlines


class CustomDataset(Dataset):
    def __init__(self, path, model_name, length = 1):
        super().__init__()
        self.preprocessing(path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.length = length

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        # lecture = self.lecture_data[idx]
        id = self.id_list[idx]
        input_data = self.tokenizer.encode_plus(text,
                                                return_tensors='pt',
                                                padding='max_length',
                                                max_length=512-self.length,
                                                truncation=True)
        input_data['input_ids'], input_data['label'] = self.random_word(input_data['input_ids'])
        input_data = {k: v.squeeze() for k, v in input_data.items()}
        input_data.update({"id": id})

        return input_data

    def preprocessing(self, path):
        # id 정보를 모아서 저장하는 예시
        self.id_list = []
        self.text_list = []
        # open 내의 디렉토리 및 파일 이름에 유의
        with jsonlines.open(path) as f:
            for line in tqdm(f):
                self.id_list.append(line['id'])
                self.text_list.append(line['text'])

    def custom_collate(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        # token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        label = torch.stack([item['label'] for item in batch])
        id = [item['id'] for item in batch]


        # Calculate max length of sequences in batch
        max_length = torch.max(torch.sum(attention_mask, dim=1)).item()

        return {'input_ids': input_ids[:, :max_length],
                # 'token_type_ids' : token_type_ids[:, :max_length],
                'attention_mask': attention_mask[:, :max_length],
                'label': label[:, :max_length],
                'id' : id}

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens[0]):
            label = tokens[0][i].detach().clone()
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[0][i] = self.tokenizer.mask_token_id # 103

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[0][i] = random.randrange(len(self.tokenizer.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[0][i] = token
                output_label.append(label)

            else:
                tokens[0][i] = token
                output_label.append(self.tokenizer.pad_token_id)

        return torch.LongTensor(tokens), torch.LongTensor(output_label)

if __name__ == "__main__":
    dataset = CustomDataset('crawled_all_dataset_0613_v4.jsonl', 'klue/roberta-large')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=dataset.custom_collate)

    for i in dataloader:
        print({k: v for k, v in i.items()})
        break

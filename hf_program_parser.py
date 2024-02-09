# using huggingface transformers to write program parser
# train on AGQA2, and inference on other video qa dataset

import os
import sys
import json
import pickle
import argparse
import logging
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, T5ForConditionalGeneration, T5Config
from transformers import TrainingArguments, Trainer

sys.path.append('../')
from tqdm import tqdm
from utils.program_parser import program_is_valid


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
device = torch.device('cuda:0')
# os.environ['WANDB_DISABLED'] = 'true'


class AGQAProgramDataset(Dataset):
    def __init__(self, data_path, data_length=None):
        self.data = pickle.load(open(data_path, 'rb'))
        if data_length is not None and len(self.data) > data_length:
            self.data = random.sample(self.data, data_length)
        # self.tokenizer = tokenizer
        logging.info('dataset length: %d', len(self))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data[index]['question']
        program = ' '.join(self.data[index]['nmn_program'])
        return question, program


class Collator:
    def __init__(self, tokenizer, max_length=128, has_program=False) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_program = has_program
        self.debug = True

    def __call__(self, examples):
        if self.debug:
            logging.info('examples: {}'.format(examples))
        question_list = [e[0] for e in examples]
        model_inputs = self.tokenizer(question_list, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        if self.has_program:
            program_list = [e[1] for e in examples]
            model_outputs = self.tokenizer(program_list, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            model_inputs['labels'] = model_outputs['input_ids']
            if self.debug:
                logging.info('model_inputs: {}'.format(model_inputs))
                self.debug = False
            return model_inputs
        else:
            question_ids = [e[1] for e in examples]
            if self.debug:
                logging.info('model_inputs: {}'.format(model_inputs))
                self.debug = False
            return model_inputs, question_ids


class AGQAQuestionDataset(Dataset):
    def __init__(self, data_path, data_length=None):
        self.data = pickle.load(open(data_path, 'rb'))
        logging.info('dataset length: %d', len(self))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['question'], self.data[index]['qa_id']


class StarQuestionDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))
        logging.info('dataset length: %d', len(self))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['question'], self.data[index]['question_id']


class NextqaQuestionDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        logging.info('dataset length: %d', len(self))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.loc[index, 'question'], str(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, choices=['train', 'test', 'check_valid'])
    parser.add_argument('--train_data_path', type=str, default='/scratch/nlp/wangyueqian/AGQA/AGQA2_balanced_lite/train_balanced.pkl')
    parser.add_argument('--valid_data_path', type=str, default='/scratch/nlp/wangyueqian/AGQA/AGQA2_balanced_lite/valid_balanced.pkl')
    parser.add_argument('--valid_data_length', type=int, default=1600)
    parser.add_argument('--model_path', type=str, default='/scratch/nlp/model_ckpts/flan-t5-large')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)

    # train
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)

    # test
    parser.add_argument('--dataset', type=str, default='STAR')

    # check valid
    parser.add_argument('--program_path', type=str, default=None)

    args = parser.parse_args()
    logging.info(args)

    if args.func in ['train', 'test']:
        config = T5Config.from_pretrained(args.model_path)
        logging.info('transformer config: {}'.format(config))
        model = T5ForConditionalGeneration.from_pretrained(args.model_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path is not None else args.model_path)
        model.to(device)

    if args.func == 'train':
        train_dataset = AGQAProgramDataset(args.train_data_path)
        valid_dataset = AGQAProgramDataset(args.valid_data_path, args.valid_data_length)
        collator = Collator(tokenizer, args.max_length, has_program=True)
        training_args = TrainingArguments(
            output_dir=args.output_path,
            evaluation_strategy='steps',
            eval_steps=args.eval_interval,
            save_steps=args.eval_interval,
            num_train_epochs=5,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(args.output_path, 'logs'),
            logging_steps=100,
            save_total_limit=3,
            disable_tqdm=True,
            report_to='tensorboard',         # disable wandb
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collator,
        )
        trainer.train()

    elif args.func == 'test':
        if args.dataset == 'AGQA':
            test_dataset = AGQAQuestionDataset(args.valid_data_path)
        if args.dataset == 'STAR':
            test_dataset = StarQuestionDataset(args.valid_data_path)
        elif args.dataset == 'MSRVTT':
            test_dataset = StarQuestionDataset(args.valid_data_path)        # they have same format 
        elif args.dataset == 'NEXTQA':
            test_dataset = NextqaQuestionDataset(args.valid_data_path)

        collator = Collator(tokenizer, args.max_length, has_program=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

        generation_config = GenerationConfig.from_pretrained(args.model_path)
        generation_config.num_beams = args.num_beams
        generation_config.num_return_sequences = args.num_beams
        generation_config.do_sample = False
        generation_config.early_stopping = True
        generation_config.max_new_tokens = args.max_length
        logging.info('generation_config: {}'.format(generation_config))

        with torch.no_grad(), open(args.output_path, 'w') as f_out:
            global_id = 0
            for batch, question_ids in test_dataloader:
                batch.to(device)
                model_output = model.generate(**batch, generation_config=generation_config, return_dict_in_generate=True)
                # logging.info('model generated: {}'.format(model_output))
                decoded_text = tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True)
                decoded_question = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                for i in range(len(decoded_question)):
                    for j in range(args.num_beams):
                        f_out.write('%s\t%s\t%s\n' % (question_ids[i], decoded_question[i], decoded_text[i*args.num_beams+j]))
                    global_id += 1

    elif args.func == 'check_valid':
        total = 0
        valid_programs = set()
        with open(args.program_path, 'r') as f:
            for line in tqdm(f.readlines()):
                try:
                    id_, question, program = line.strip().split('\t')
                except:
                    print(line)
                    continue
                total += 1 / args.num_beams
                if id_ in valid_programs:
                    continue
                if program_is_valid(program.split(' ')):
                    valid_programs.add(id_)
        logging.info('total: %d, valid: %d, invalid: %d, valid_rate: %.4lf' % (total, len(valid_programs), total-len(valid_programs), len(valid_programs)/total))

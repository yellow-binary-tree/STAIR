# finetune BERT baseline
import os
import sys
import datetime
import pickle
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification, BertConfig, BertTokenizer,
    GPT2Config, GPT2Tokenizer,
    LlamaTokenizer, LlamaForCausalLM,
    AutoTokenizer, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)

from video_nmn.dataset import format_print, to_device
from video_nmn.dataset import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT
from video_nmn.args import get_args
from utils.program_parser import get_childrens_and_parents, stat_module_levels
# from modeling_gpt2 import GPT2Model, GPT2LMHeadModel

from VideoGPT2 import LMHeadModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AGQADataset:
    def __init__(self, args, split, filter_output_path=None, prompt='%s'):
        self.args = args
        data_filename = {'train': args.train_filename, 'valid': args.valid_filename, 'test': args.test_filename}[split]
        self.data = pickle.load(open(data_filename, 'rb'))
        self.appearance_path = args.rgb_path
        self.max_video_length = args.max_video_length
        self.tokenizer_max_length = args.tokenizer_max_length
        self.prompt = prompt

        self.gpt_max_per_filter_module = args.gpt_max_per_filter_module
        self.gpt_max_filter_output_list_length = args.gpt_max_filter_output_list_length

        # data filter for generalization study
        if args.novel_comp is not None:
            print('loading data only with novel_comp = %d!' % args.novel_comp)
            self.data = [d for d in self.data if d['novel_comp'] == args.novel_comp]
        if args.more_steps is not None:
            print('loading data only with more_steps = %d!' % args.more_steps)
            self.data = [d for d in self.data if d['more_steps'] == args.more_steps]

        if args.debug:
            self.data = random.sample(self.data, 256)
            print('DEBUG MODE, random sampling only %d data in dataset' % len(self.data))

        print('loading answer vocab')
        self.answer_vocab = json.load(open(args.vocab_filename))
        int_key_dict = {}         # json turns all int keys to string keys, we need to change them back to int
        for key, word in self.answer_vocab['id2word'].items():
            int_key_dict[int(key)] = word
        self.answer_vocab['id2word'] = int_key_dict

        assert 'word2id' in self.answer_vocab
        assert 'id2word' in self.answer_vocab
        assert len(self.answer_vocab['id2word']) == len(self.answer_vocab['word2id'])
        assert [self.answer_vocab['id2word'][i] for i in range(4)] == ['yes', 'no', 'before', 'after']

        print('loading video features')
        used_video_ids = list(set(d['video_id'] for d in self.data))
        self.load_video_features_from_disk(used_video_ids)

        print('loading BERT tokenizer')
        if args.lm_model == 'Llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(args.bert_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

        if filter_output_path is not None:
            print('load filter output if NMN as GPT prompt')
            filter_otput_split = 'debug' if args.debug else split
            self.filter_output_dict = dict()

            if '%d' in filter_output_path:
                i = -1
                while True:
                    i += 1
                    if os.path.exists(filter_output_path % (filter_otput_split, i)):
                        filter_output_dict_delta = pickle.load(open(filter_output_path % (filter_otput_split, i), 'rb'))
                        self.filter_output_dict = {**self.filter_output_dict, **filter_output_dict_delta}
                    else:
                        break
            else:
                self.filter_output_dict = pickle.load(open(filter_output_path % (filter_otput_split), 'rb'))

            print('length of filter output examples:', len(self.filter_output_dict))

        else:
            self.filter_output_dict = None

        print('DATASET: finished loading %s dataset! loaded %d examples' % (split, len(self)))
        print('5 random examples in dataset:')
        for example_i in random.sample(range(len(self)), 5):
            batch = self[example_i]
            for key, value in batch.items():
                print(key, format_print(value))

    def __len__(self):
        return len(self.data)

    def load_video_features_from_disk(self, used_video_ids):
        print('loading %d video features from disc' % len(used_video_ids))
        self.video_feats = dict()
        for fname in os.listdir(self.appearance_path):
            video_id = fname.split('.')[0]
            if video_id in used_video_ids:
                video_feat = np.load(os.path.join(self.appearance_path, fname))
                select_idx = np.arange(start=0, stop=video_feat.shape[0], step=2)
                video_feat = video_feat[select_idx, :]
                if video_feat.shape[0] > self.max_video_length:
                    video_feat = video_feat[:self.max_video_length]
                self.video_feats[video_id] = torch.tensor(video_feat).squeeze()

    def __getitem__(self, idx):
        json_data = self.data[idx]
        video_id = json_data['video_id']
        video_features = self.video_feats[video_id]
        video_length = video_features.size(0)
        ret = self.get_text_data(idx)
        ret['video_feat'] = video_features
        ret['video_length'] = video_length
        return ret

    def answer_vocab_length(self):
        return len(self.answer_vocab['word2id'])

    def get_text_data(self, idx):
        json_data = self.data[idx]
        answer = torch.tensor(self.answer_vocab['word2id'].get(json_data['answer'], self.answer_vocab['word2id'].get('<UNK>')))
        question_text = self.prompt % json_data['question']

        if self.filter_output_dict is not None:
            # add nmn finter output as prompt
            filter_output_dict = self.filter_output_dict.get(json_data['qa_id'], dict())

            if self.args.gpt_filter_output_by_level:
                filter_output_list = [(level, kw, ans_list) for (level, kw, ans_list) in filter_output_dict.values() if level <= args.gpt_filter_output_by_level]
            else:
                filter_output_list = list(filter_output_dict.values())
            filter_output_list.sort(key=lambda x: -x[0])        # from lower level to higher level
            filter_output_texts = list()
            for _, kw, ans_list in filter_output_list:
                for ans in ans_list[:self.gpt_max_per_filter_module]:
                    filter_output_texts.append('%s %s.' % (kw, ans))

            if not self.args.gpt_filter_output_by_level and len(filter_output_texts) > self.gpt_max_filter_output_list_length:
                filter_output_texts = filter_output_texts[:self.gpt_max_filter_output_list_length]
            if filter_output_texts:
                question_text = ' '.join(filter_output_texts) + ' ' + question_text

        token_ids = self.tokenizer(
            question_text, return_tensors='pt', max_length=self.tokenizer_max_length, truncation=True, padding=False, add_special_tokens=False
        )['input_ids'][0]
        answer_token_ids = self.tokenizer(
            json_data['answer'], return_tensors='pt', max_length=self.tokenizer_max_length, truncation=True, padding=False, add_special_tokens=False
        )['input_ids'][0]
        answer_token_ids = torch.cat([
            answer_token_ids, torch.tensor([self.tokenizer.eos_token_id])
        ])

        return {'token_ids': token_ids, 'answer': answer, 'answer_token_ids': answer_token_ids, 'question_text': question_text, 'qa_id': json_data['qa_id'], 'answer_text': json_data['answer']}



def collate_fn(examples):
    ret = {}
    for key in examples[0]:
        if key in ['answer']:
            ret[key] = torch.stack([e[key] for e in examples])
        else:
            ret[key] = [e[key] for e in examples]
    return ret


@torch.no_grad()
def evaluate(args, dataloader, model, linear_model, classifier=None, print_progress=True, preds_file=None):
    loss_list, acc_list, preds_golds_list = list(), list(), {'preds': [], 'golds': [], 'qa_ids': []}
    tokenizer = dataloader.dataset.tokenizer
    pad_token_id = tokenizer.pad_token_id

    model.eval()
    linear_model.eval()
    if classifier is not None:
        classifier.eval()

    for i, batch in enumerate(dataloader):
        if print_progress and i % 100 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time, 'test progress: %d batches' % (i))

        batch = to_device(batch, device)

        # build bert model input
        bert_inputs_embeds, bert_token_type_ids, bert_position_ids, reply_attention_mask, video_attention_mask, output_pos_list, video_lengths_list = \
            get_gpt_input(batch, model, linear_model, tokenizer, args.max_video_length, args.lm_model)

        lm_labels = output_pos_list

        gpt_output = model(input_embs=bert_inputs_embeds, token_type_ids=bert_token_type_ids, labels=(lm_labels, None), 
                            attention_mask=[torch.zeros_like(reply_attention_mask), reply_attention_mask])
        loss, lm_logits = gpt_output[0], gpt_output[1]

        loss_list.append(loss.detach().cpu().item())
        for lm_label, lm_logit, qa_id in zip(lm_labels[..., 1:], lm_logits, batch['qa_id']):
            answer_pos = ((lm_label != -1) & (lm_label != pad_token_id)).nonzero().squeeze(1)     # not given token and not eos token
            ans = lm_label[answer_pos]
            pred = lm_logit[answer_pos].max(dim=1).indices
            acc_list.append((pred == ans).all().cpu().item())
            preds_golds_list['preds'].append(tokenizer.decode(pred, skip_special_tokens=True))
            preds_golds_list['golds'].append(tokenizer.decode(ans, skip_special_tokens=True))
            preds_golds_list['qa_ids'].append(qa_id)

    print('evaluation finished, evaluated on %d examples' % len(acc_list))
    if preds_file is not None:
        json.dump(preds_golds_list, open(preds_file, 'w'))

    model.train()
    linear_model.train()
    if classifier is not None:
        classifier.train()

    return sum(acc_list) / len(acc_list), sum(loss_list) / len(loss_list)


def save_model(folder, model, linear_model, classifier=None):
    model.save_pretrained(folder)
    if linear_model is not None:
        torch.save(linear_model.state_dict(), os.path.join(folder, 'linear_model.bin'))
    if classifier is not None:
        torch.save(classifier.state_dict(), os.path.join(folder, 'classifier.bin'))


def main(args):
    print(args)
    filter_output_path = args.gpt_filter_result_path
    prompt = 'Question: %s Answer:' if args.lm_model == 'Llama' else '%s'
    if not args.gpt_test:
        # load train dataset
        train_dataset = AGQADataset(args, split='train', filter_output_path=filter_output_path, prompt=prompt)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

        valid_dataset = AGQADataset(args, split='valid', filter_output_path=filter_output_path, prompt=prompt)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    else:
        # load test dataset
        test_dataset = AGQADataset(args, split='test', filter_output_path=filter_output_path, prompt=prompt)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        train_dataset = test_dataset        # may be used later

    # load model
    bert_path = args.model_ckpt if args.model_ckpt else args.bert_path
    print('LOADING MODEL from', bert_path)
    config = AutoConfig.from_pretrained(bert_path)
    config.video_feature_dim = args.video_size
    config.llm_lora = args.llm_lora
    model = LMHeadModel(config=config, model_path=bert_path, lm_model=args.lm_model)
    model.resize_token_embeddings(len(train_dataset.tokenizer))
    classifier = None

    linear_model = nn.Linear(args.video_size, config.hidden_size)
    model.cuda()
    linear_model.cuda()

    if args.gpt_test:
        valid_acc, valid_loss = evaluate(args, test_dataloader, model, linear_model, classifier, print_progress=True,
                                         preds_file=os.path.join(args.output, args.result_filename))
        print('valid loss = %.4lf, valid acc = %.4lf' % (valid_loss, valid_acc))

    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_epochs
        )

        # load tensorboard
        tensorboard_dir = os.path.join(args.output, 'runs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        tb_writer = SummaryWriter(tensorboard_dir)

        # Train!
        global_steps, best_acc = 0, 0.
        running_loss = list()
        for epoch_i in range(args.num_epochs):
            for batch in train_dataloader:
                global_steps += 1
                batch = to_device(batch, device)

                bert_inputs_embeds, bert_token_type_ids, bert_position_ids, padding_attention_mask, video_attention_mask, output_pos_list, video_lengths_list = \
                    get_gpt_input(batch, model, linear_model, train_dataset.tokenizer, args.max_video_length, args.lm_model)

                lm_labels = output_pos_list

                gpt_output_reply = model(input_embs=bert_inputs_embeds, token_type_ids=bert_token_type_ids, labels=(lm_labels, None),
                                        attention_mask=[torch.zeros_like(padding_attention_mask), padding_attention_mask])
                loss = gpt_output_reply[0]

                if args.gpt_video_loss_weight != 0:
                    gpt_output_video = model(input_embs=bert_inputs_embeds, token_type_ids=bert_token_type_ids, labels=(lm_labels, batch['video_feat']),
                                            attention_mask=[video_attention_mask, padding_attention_mask], mode='video', video_lengths=video_lengths_list)
                    loss += gpt_output_video[0] * args.gpt_video_loss_weight

                running_loss.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if global_steps % args.report_interval == 0:
                    average_loss = sum(running_loss) / len(running_loss)
                    running_loss = list()
                    tb_writer.add_scalar('loss/loss', average_loss, global_steps)
                    tb_writer.add_scalar('lr/lr', scheduler.get_last_lr()[0], global_steps)

                if global_steps % args.evaluate_interval == 0:
                    valid_acc, valid_loss = evaluate(args, valid_dataloader, model, linear_model, classifier)
                    print('valid loss = %.4lf, valid acc = %.4lf' % (valid_loss, valid_acc))
                    tb_writer.add_scalar('valid/loss', valid_loss, global_steps)
                    tb_writer.add_scalar('valid/acc', valid_acc, global_steps)

                    if valid_acc > best_acc:
                        save_model(os.path.join(args.output, 'best_model'), model, linear_model, classifier)
                        print('saving model: prev valid acc = %.4lf, new valid acc = %.4lf' % (best_acc, valid_acc))
                        best_acc = valid_acc


def get_gpt_input(batch, model, linear_model, tokenizer, video_max_length=None, lm_model='VideoGPT'):
    token_ids_list, video_lengths_list, video_feat_list = batch['token_ids'], batch['video_length'], batch['video_feat']
    if video_max_length is None:
        video_max_length = max(video_lengths_list)
    token_lengths_list = [t.size(0) for t in token_ids_list]
    token_max_length = max(token_lengths_list)
    bert_inputs_embeds_list, bert_token_type_ids_list, bert_position_ids_list, bert_attention_masks_list, bert_lm_labels_list, output_pos_list = \
        list(), list(), list(), list(), list(), list()
    pad_token_id = tokenizer.pad_token_id

    pad_embed = model.transformer.get_input_embeddings()(torch.tensor([pad_token_id], device=device))
    answer_token_ids_list = batch['answer_token_ids']
    answer_lengths_list = [t.size(0) for t in answer_token_ids_list]
    answer_max_length = max(answer_lengths_list)
    padding_attention_masks_list, video_attention_masks_list = list(), list()

    for video_feat, video_length, token_idx, token_length, answer_idx, answer_length in zip(
            video_feat_list, video_lengths_list, token_ids_list, token_lengths_list, answer_token_ids_list, answer_lengths_list):
        token_embeddings = model.transformer.get_input_embeddings()(token_idx)
        video_embeddings = model.video_ff(video_feat[:video_length])
        video_padding_embeddings = pad_embed.repeat(video_max_length - video_length, 1)
        answer_embeddings = model.transformer.get_input_embeddings()(answer_idx)
        # padding_embeddings = pad_embed.repeat(video_max_length + token_max_length + answer_max_length - video_length - token_length - answer_length, 1)
        token_padding_embeddings = pad_embed.repeat(token_max_length + answer_max_length - token_length - answer_length, 1)
        # bert_inputs_embeds_list.append(torch.cat([video_embeddings, token_embeddings, answer_embeddings, padding_embeddings], dim=0))
        bert_inputs_embeds_list.append(torch.cat([video_embeddings, video_padding_embeddings, token_embeddings, answer_embeddings, token_padding_embeddings], dim=0))

        if lm_model == 'Llama':
            token_type_ids = torch.cat([
                # torch.ones(video_length, device=device, dtype=torch.long) * tokenizer.convert_tokens_to_ids('<video>'),
                # torch.ones(video_max_length + token_max_length + answer_max_length - video_length, device=device, dtype=torch.long) * tokenizer.convert_tokens_to_ids('<cap>'),
                torch.ones(video_max_length, device=device, dtype=torch.long) * tokenizer.convert_tokens_to_ids('<video>'),
                torch.ones(token_max_length + answer_max_length, device=device, dtype=torch.long) * tokenizer.convert_tokens_to_ids('<cap>'),
            ])
        else:
            token_type_ids = torch.zeros(video_max_length + token_max_length + answer_max_length, device=device, dtype=torch.long)      # this will not be used anyway

        position_ids = torch.arange(0, video_max_length + token_max_length + answer_max_length, device=device, dtype=torch.long)

        bert_token_type_ids_list.append(token_type_ids)
        bert_position_ids_list.append(position_ids)

        video_attention_mask = torch.cat([
            # torch.ones(video_length, device=device, dtype=torch.long),
            # torch.zeros(video_max_length + token_max_length + answer_max_length - video_length, device=device, dtype=torch.long)    # unmask all video tokens
            torch.ones(video_max_length, device=device, dtype=torch.long),
            torch.zeros(token_max_length + answer_max_length, device=device, dtype=torch.long)    # unmask all video tokens
        ])

        padding_attention_mask = torch.cat([
            torch.ones(video_length, device=device, dtype=torch.long),
            torch.zeros(video_max_length - video_length, device=device, dtype=torch.long),
            torch.ones(token_length + answer_length, device=device, dtype=torch.long),
            torch.zeros(token_max_length + answer_max_length - token_length - answer_length, device=device, dtype=torch.long)
        ])
        video_attention_masks_list.append(video_attention_mask)
        padding_attention_masks_list.append(padding_attention_mask)

        lm_labels = torch.cat([
            # torch.ones(video_length + token_length, device=device, dtype=torch.long) * -1,      # FOR VIDEOGPT
            torch.ones(video_max_length + token_length, device=device, dtype=torch.long) * -1,      # FOR VIDEOGPT
            answer_idx,
            # torch.ones(video_max_length + token_max_length + answer_max_length - video_length - token_length - answer_length, device=device, dtype=torch.long) * -1,
            torch.ones(token_max_length + answer_max_length - token_length - answer_length, device=device, dtype=torch.long) * -1,
        ])
        bert_lm_labels_list.append(lm_labels)

    bert_inputs_embeds, bert_token_type_ids, bert_position_ids, padding_attention_masks, video_attention_mask, bert_lm_labels = \
        torch.stack(bert_inputs_embeds_list), torch.stack(bert_token_type_ids_list), torch.stack(bert_position_ids_list), torch.stack(padding_attention_masks_list), torch.stack(video_attention_masks_list), torch.stack(bert_lm_labels_list)
    return bert_inputs_embeds, bert_token_type_ids, bert_position_ids, padding_attention_masks, video_attention_mask, bert_lm_labels, video_lengths_list


if __name__ == '__main__':
    args = get_args()
    main(args)

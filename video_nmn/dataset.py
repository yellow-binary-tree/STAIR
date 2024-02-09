
import os
import json
import pickle
from collections import Counter
import random
import copy
import h5py

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer

from utils.program_parser import parse_program
from utils.program_parser import nary_mappings as NARY_MAPPINGS
WORDS_TO_KEEP = {'forward', 'backward', 'while', 'between', 'before', 'after', 'max', 'min', 'start', 'end', 'video'}
ALL_KWS = WORDS_TO_KEEP | set(NARY_MAPPINGS.keys())

# GPT2 generation
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}


class AGQADataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.debug = args.debug
        self.split = split
        self.appearance_path, self.motion_path = args.rgb_path, args.flow_path

        self.video_secs = json.load(open(args.video_secs_path))
        if os.path.isfile(self.appearance_path):
            self.str2num = json.load(open(args.str2num_path))

        data_filename = {'train': args.train_filename, 'valid': args.valid_filename, 'test': args.test_filename}[split]
        self.data = list()
        data_ = pickle.load(open(data_filename, 'rb'))

        # load all data
        if split in ['train', 'valid']:     # remove data with missing values when training and validating
            self.data = list()
            for i, data in enumerate(data_):
                if data['sg_res_by_step'] is None:
                    data['sg_res_by_step'] = {}     # here we use the answer as supervision only
                if (None, None) in data['nmn_program_span_by_word'].values():
                    continue
                self.data.append(data)
        else:
            self.data = data_

        # data filter for generalization study
        if args.novel_comp is not None:
            print('loading data only with novel_comp = %d!' % args.novel_comp)
            self.data = [d for d in self.data if d['novel_comp'] == args.novel_comp]
        if args.more_steps is not None:
            print('loading data only with more_steps = %d!' % args.more_steps)
            self.data = [d for d in self.data if d['more_steps'] == args.more_steps]

        if self.debug:
            self.data = random.sample(self.data, 256)
            print('DEBUG MODE, random sampling only %d data in dataset' % len(self.data))

        # create or load answer vocab
        if not os.path.exists(args.vocab_filename):
            print('creating answer vocab')
            answer_counter = Counter([data['answer'] for data in self.data])
            answer_vocab = ['yes', 'no', 'before', 'after']
            given_choices = set(answer_vocab)
            for ans, _ in sorted(answer_counter.items(), key=lambda x: -x[1]):
                if ans not in given_choices:
                    answer_vocab.append(ans)
            answer_vocab.append('<UNK>')
            self.answer_vocab = {}
            self.answer_vocab['word2id'] = {word: i for i, word in enumerate(answer_vocab)}
            self.answer_vocab['id2word'] = {i: word for i, word in enumerate(answer_vocab)}
            json.dump(self.answer_vocab, open(args.vocab_filename, 'w'))
        else:
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

        self.max_video_length = args.max_video_length
        assert args.max_video_length >= 2, 'why do you set max video length so small?'

        self.load_glove(args.glove_filename)
        self.word_embedding_size = self.word_embeddings['the'].size

        # random shuffled video features
        used_video_ids = list(set(d['video_id'] for d in self.data))
        self.shuffle_video = args.shuffle_video
        if self.shuffle_video:
            print('random shuffling video ids for ablation study!')
            shuffled_ids = list(range(len(used_video_ids)))
            random.shuffle(shuffled_ids)
            self.shuffle_video_mapping = {used_video_ids[i]: used_video_ids[shuffled_ids[i]] for i in range(len(used_video_ids))}

        # load all video feature from disk into memory
        self.load_video_features_from_disk(used_video_ids)

        print('DATASET: finished loading %s dataset! loaded %d examples' % (split, len(self)))
        print('example data in dataset:')
        batch = self[0]
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.numel() > 100:
                value = value.size()
            print(key, value)

    def __len__(self):
        return len(self.data)

    def get_question_feature(self, json_data):
        text_inputs = self.embed_sent(json_data['question'])
        prog_str_to_question_tokens = json_data['nmn_program_span_by_word']
        return text_inputs, prog_str_to_question_tokens

    def load_video_features_from_disk(self, used_video_ids):
        print('loading %d video features from disk' % len(used_video_ids))
        self.video_feats = dict()
        if os.path.isdir(self.appearance_path):
            for fname in os.listdir(self.appearance_path):
                video_id = fname.split('.')[0]
                if video_id in used_video_ids:
                    video_feat = np.load(os.path.join(self.appearance_path, fname))
                    select_idx = np.arange(start=0, stop=video_feat.shape[0], step=2)
                    video_feat = video_feat[select_idx, :]
                    if video_feat.shape[0] > self.max_video_length:
                        video_feat = video_feat[:self.max_video_length]
                    self.video_feats[video_id] = torch.tensor(video_feat).squeeze()

        elif os.path.isfile(self.appearance_path):
            f_feat = h5py.File(self.appearance_path)
            id2id = {id_: i for i, id_ in enumerate(f_feat['ids'][()])}
            for video_id, id_ in self.str2num.items():
                if video_id in used_video_ids:
                    video_feat = f_feat['resnet_features'][id2id[id_]]
                    if video_feat.shape[0] > self.max_video_length:
                        video_feat = video_feat[:self.max_video_length]
                    video_feat = torch.tensor(video_feat).mean(dim=1)
                    self.video_feats[video_id] = video_feat

        else:
            raise ValueError('appearance path not given!')

        # concat motion feat to video feats, if given
        if self.motion_path is not None:
            if os.path.isdir(self.motion_path):
                pass        # TODO
            elif os.path.isfile(self.motion_path):
                f_feat = h5py.File(self.motion_path)
                id2id = {id_: i for i, id_ in enumerate(f_feat['ids'][()])}
                for video_id, id_ in self.str2num.items():
                    if video_id in used_video_ids:
                        video_feat = f_feat['resnext_features'][id2id[id_]]
                        if video_feat.shape[0] > self.max_video_length:
                            video_feat = video_feat[:self.max_video_length]
                        video_feat = torch.tensor(video_feat)
                        self.video_feats[video_id] = torch.cat([self.video_feats[video_id], video_feat], dim=-1)

    def __getitem__(self, idx):
        json_data = self.data[idx]
        text_inputs, prog_str_to_question_tokens = self.get_question_feature(json_data)

        # load video features
        if self.shuffle_video:
            video_id = self.shuffle_video_mapping[json_data['video_id']]
        else:
            video_id = json_data['video_id']
        video_features = self.video_feats[video_id]
        video_length = video_features.size(0)

        # load answer_id
        answer = torch.tensor(self.answer_vocab['word2id'].get(json_data['answer'], self.answer_vocab['word2id'].get('<UNK>')))

        if self.split == 'test':
            # for test set, we only need to provide necessary inputs and answer
            ret_dict = {
                'question': text_inputs, 'answer': answer, 'video_features': video_features,
                'prog_str_to_question_tokens': prog_str_to_question_tokens,
                'nmn_program_list': json_data['nmn_program'], 'nmn_program_idx': json_data['nmn_program_idx'],
                'qa_id': json_data['qa_id'], 'question_raw': json_data['question']
            }
            return ret_dict

        else:
            # for train/valid set, we need to provide the intermediate results
            src_length = self.video_secs[video_id] * 3
            sg_res_by_step = dict()
            for key, value in json_data['sg_res_by_step'].items():
                if isinstance(value, (tuple, list)) and len(value) >= 1:
                    if isinstance(value[0], float):
                        value = frame_interval_change_fps(value, src_length, video_length)
                    if isinstance(value[0], tuple) and isinstance(value[0][0], float):
                        value = tuple([frame_interval_change_fps(v, src_length, video_length) for v in value])
                if isinstance(value, dict) and isinstance(list(value.values())[0], tuple) and isinstance(list(value.values())[0][0], float):
                    value = {k: frame_interval_change_fps(v, src_length, video_length) for k, v in value.items()}
                sg_res_by_step[key] = value

            new_sg_res_by_step = dict()
            for key, value in sg_res_by_step.items():
                if isinstance(value, str):
                    new_sg_res_by_step[key] = [(value, self.embed_sent(value))]
                elif isinstance(value, list) and len(value) and isinstance(value[0], str):
                    new_sg_res_by_step[key] = [(v, self.embed_sent(v)) for v in value]
                else:
                    new_sg_res_by_step[key] = value
            sg_res_by_step = new_sg_res_by_step

            ret_dict = {
                # input
                'question': text_inputs, 'answer': answer, 'video_features': video_features,
                'prog_str_to_question_tokens': prog_str_to_question_tokens,
                # programs
                'nmn_program_list': json_data['nmn_program'], 'nmn_program_idx': json_data['nmn_program_idx'],
                'sg_program_list': json_data['sg_program'], 'sg_res_by_step': sg_res_by_step,
                # for debugging
                'qa_id': json_data['qa_id'], 'question_raw': json_data['question']
            }
            return ret_dict

    def load_glove(self, glove_filename):
        if glove_filename.endswith('.pkl'):
            self.word_embeddings = pickle.load(open(glove_filename, 'rb'))
        else:
            self.word_embeddings = dict()
            for i, line in enumerate(open(glove_filename)):
                if i == 0:
                    length, shape = map(int, line.split(' '))
                else:
                    line_split = line.split(' ')
                    word, feat = line_split[0], list(map(float, line_split[1:]))
                    self.word_embeddings[word] = np.array(feat)

    def embed_sent(self, sent):
        if isinstance(sent, str):
            words = word_tokenize(sent.lower())
        else:
            assert isinstance(sent, list) and isinstance(sent[0], str)
            words = [s.lower() for s in sent]
        embeddings = [self.word_embeddings[w] if w in self.word_embeddings else np.random.rand(self.word_embedding_size) for w in words] 
        return torch.tensor(embeddings, dtype=torch.float32)

    def answer_vocab_length(self):
        return len(self.answer_vocab['word2id'])


def frame_interval_change_fps(interval_tuple, src_length, tgt_length):
    start = interval_tuple[0] / src_length * tgt_length
    end = interval_tuple[1] / src_length * tgt_length
    return (start, end)


class STARDataset(AGQADataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.appearance_path, self.motion_path = args.rgb_path, args.flow_path
        self.use_prog_word_embeddings = self.args.use_prog_word_embeddings 
        self.tokenizer_max_length = args.tokenizer_max_length

        # load data and program from files
        data_filename = {'train': args.train_filename, 'valid': args.valid_filename, 'test': args.test_filename}[split]
        data_ = pickle.load(open(data_filename, 'rb'))

        # if this is train/valid set, discard the examples with invalid program;
        # if this is test set, use a empty program
        self.data = list()
        if split in ['train', 'valid']:
            for data in data_:
                if data['nmn_program']:         # and data['nmn_program_span_by_word']: #  感觉不需要判断后面这个是否存在
                    if isinstance(data['answer'], str):
                        data['answer_id'] = [i for i, c in enumerate(data['choices']) if c['choice'] == data['answer']][0]
                    else:
                        data['answer_id'] = data['answer']
                    data['question'] = data['question'].replace('/', ' ')
                    self.data.append(data)

        elif split == 'test':
            for data in data_:
                data['question'] = data['question'].replace('/', ' ')
                self.data.append(data)

        if split == 'train' and args.dataset == 'STAR':
            self.sample_more_candidates_by_question_type()

        self.load_glove(args.glove_filename)
        self.word_embedding_size = self.word_embeddings['the'].size

        # load videos
        self.max_video_length = args.max_video_length
        used_video_ids = list(set(d['video_id'] for d in self.data))
        self.load_video_features_from_disk(used_video_ids)
        self.video_secs = json.load(open(args.video_secs_path))

        print('DATASET: finished loading %s dataset! loaded %d examples' % (split, len(self)))
        print('example data in dataset:')
        batch = self[0]
        for key, value in batch.items():
            print(key, format_print(value))

    def sample_more_candidates_by_question_type(self, candidate_num=10):
        answers_by_question_type = {key: set() for key in ['Interaction', 'Sequence', 'Prediction', 'Feasibility']}
        for data in self.data:
            answers_by_question_type[data['question_id'].split('_')[0]].add(data['answer'])

        for data in self.data:
            question_type = data['question_id'].split('_')[0]
            answers = copy.deepcopy(answers_by_question_type[question_type])
            if data['answer'] in answers:
                answers.remove(data['answer'])
            new_negs = random.sample(answers, k=candidate_num)
            num_negs_in_data = len(data['choices'])
            for i, neg in enumerate(new_negs):
                data['choices'].append({'choice': neg, 'choice_id': i + num_negs_in_data})

    def get_video_feat(self, video_id, start=None, end=None):
        video_features = self.video_feats[video_id]
        if start is None and end is None:
            return video_features
        src_length = self.video_secs[video_id]
        video_length = video_features.size(0)
        start, end = map(int, frame_interval_change_fps((start, end), src_length, video_length))
        return video_features[start:end]

    def get_candidates_feat(self, candidates):
        return [self.embed_sent(candidate) for candidate in candidates], None

    def __getitem__(self, idx):
        json_data = self.data[idx]
        video_features = self.get_video_feat(json_data['video_id'], json_data.get('start', None), json_data.get('end', None))
        lstm_text_inputs, bert_text_inputs, prog_str_to_question_tokens = self.get_question_feature(json_data)
        candidates_raw = [c['choice'].replace('/', ' ') for c in json_data['choices']]
        lstm_candidate_inputs, bert_candidate_inputs = self.get_candidates_feat(candidates_raw)

        ret_dict = {
            'lstm_question': lstm_text_inputs, 'bert_question': bert_text_inputs,
            'lstm_candidates': lstm_candidate_inputs, 'bert_candidates': bert_candidate_inputs, 'num_candidates': len(candidates_raw),
            'video_features': video_features, 'nmn_program_list': json_data['nmn_program'],
            'prog_str_to_question_tokens': prog_str_to_question_tokens,
            'qa_id': json_data['question_id'],
            'question_raw': json_data['question'].replace('/', ' '), 'candidates_raw': candidates_raw,
        }

        # as many words in nmn_program can't find span in question, we need to encode them
        prog_word_embeddings = dict()
        if self.use_prog_word_embeddings:
            if json_data['nmn_program_span_by_word'] is not None:
                for prog_idx, span in json_data['nmn_program_span_by_word'].items():
                    if None in span:
                        prog_word_embeddings[prog_idx] = self.embed_sent(json_data['nmn_program'][prog_idx].replace('_', ' ').replace('/', ' '))
            ret_dict['prog_word_embeddings'] = prog_word_embeddings

        if self.split in ['train', 'valid']:
            ret_dict['answer'] = torch.tensor(json_data['answer_id'])
        return ret_dict


class MSRVTTDataset(STARDataset):
    def __init__(self, args, split):
        self.args = args
        self.debug = args.debug
        self.split = split
        self.tokenizer_max_length = args.tokenizer_max_length
        self.appearance_path, self.motion_path = args.rgb_path, args.flow_path
        self.use_prog_word_embeddings = self.args.use_prog_word_embeddings 

        data_filename = {'train': args.train_filename, 'valid': args.valid_filename, 'test': args.test_filename}[split]
        data_ = pickle.load(open(data_filename, 'rb'))
        self.data = list()
        if split in ['train', 'valid']:     # remove data with missing values
            self.data = list()
            for i, data in enumerate(data_):
                if data['nmn_program']:
                    data['video_id'] = data['video'].replace('.mp4', '')
                    self.data.append(data)
        else:
            for i, data in enumerate(data_):
                data['video_id'] = data['video'].replace('.mp4', '')
                self.data.append(data)

        # create or load answer vocab
        if not os.path.exists(args.vocab_filename):
            print('creating answer vocab')
            answer_counter = Counter([data['answer'] for data in self.data])
            answer_vocab = []
            for ans, _ in sorted(answer_counter.items(), key=lambda x: -x[1]):
                answer_vocab.append(ans)
                if len(answer_vocab) >= args.max_vocab_length:
                    break
            answer_vocab.append('<UNK>')
            self.answer_vocab = {}
            self.answer_vocab['word2id'] = {word: i for i, word in enumerate(answer_vocab)}
            self.answer_vocab['id2word'] = {i: word for i, word in enumerate(answer_vocab)}
            json.dump(self.answer_vocab, open(args.vocab_filename, 'w'))
        else:
            print('loading answer vocab')
            self.answer_vocab = json.load(open(args.vocab_filename))
            int_key_dict = {}         # json turns all int keys to string keys, we need to change them back to int
            for key, word in self.answer_vocab['id2word'].items():
                int_key_dict[int(key)] = word
            self.answer_vocab['id2word'] = int_key_dict

            assert 'word2id' in self.answer_vocab
            assert 'id2word' in self.answer_vocab
            assert len(self.answer_vocab['id2word']) == len(self.answer_vocab['word2id'])

        self.max_video_length = args.max_video_length

        self.load_glove(args.glove_filename)
        self.word_embedding_size = self.word_embeddings['the'].size

        used_video_ids = list(set(d['video_id'] for d in self.data))
        self.load_video_features_from_disk(used_video_ids)

        print('DATASET: finished loading %s dataset! loaded %d examples' % (split, len(self)))
        print('example data in dataset:')
        batch = self[0]
        for key, value in batch.items():
            print(key, format_print(value))

    def __getitem__(self, idx):
        json_data = self.data[idx]
        video_features = self.get_video_feat(json_data['video_id'])
        lstm_text_inputs, bert_text_inputs, prog_str_to_question_tokens = self.get_question_feature(json_data)
        answer_raw = json_data['answer']
        answer = torch.tensor(self.answer_vocab['word2id'].get(json_data['answer'], self.answer_vocab['word2id'].get('<UNK>')))

        ret_dict = {
            'lstm_question': lstm_text_inputs, 'bert_question': bert_text_inputs,
            'video_features': video_features, 'answer': answer,
            'nmn_program_list': json_data['nmn_program'],
            'prog_str_to_question_tokens': prog_str_to_question_tokens,
            'qa_id': json_data['question_id'],
            'question_raw': json_data['question'], 'answer_raw': answer_raw
        }

        # as many words in nmn_program can't find span in question, we need to encode them
        prog_word_embeddings = dict()
        if self.use_prog_word_embeddings:
            if json_data['nmn_program_span_by_word'] is not None:
                for prog_idx, span in json_data['nmn_program_span_by_word'].items():
                    if None in span:
                        candidate_text = json_data['nmn_program'][prog_idx].replace('_', ' ').replace('/', ' ')
                        prog_word_embeddings[prog_idx] = self.embed_sent(candidate_text)
        ret_dict['prog_word_embeddings'] = prog_word_embeddings
        return ret_dict


def collate_fn(examples):
    return examples[0]


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [to_device(i, device) for i in obj]
    if isinstance(obj, tuple):
        return tuple([to_device(i, device) for i in obj])
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def format_print(obj):
    if isinstance(obj, PackedSequence):
        return type(obj)
    if isinstance(obj, torch.Tensor) and obj.numel() > 100:
        return obj.size()
    if isinstance(obj, list):
        return [format_print(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple([format_print(i) for i in obj])
    if isinstance(obj, dict):
        return {k: format_print(v) for k, v in obj.items()}
    return obj

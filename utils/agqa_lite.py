import os
import sys
import random
import json
import copy
import pickle
import argparse
import datetime
from itertools import combinations
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wnl = WordNetLemmatizer()

import program_parser
import scene_graphs

WORDS_TO_KEEP = {'forward', 'backward', 'while', 'between', 'before', 'after', 'max', 'min', 'start', 'end', 'video', 'relations', 'objects', 'actions'}
ALL_KWS = WORDS_TO_KEEP | program_parser.nary_mappings.keys()

rules_dict_question = {'consume': 'eat', 'consuming': 'eat', 'ate': 'eat', 'taking': 'take', 'sneezing': 'sneeze', 'drank': 'drink', 'wiping': 'wipe', 'drinking': 'drink', 'closing': 'close', 'lay': 'lie'}
rules_dict_prog = {'opening': 'open', 'closing': 'close', 'sitting on': 'sit', 'playing on': 'play', 'drinking': 'drink', 'putting down': 'put', 'consuming': 'eat'}

sg_executer = None


def get_scene_graph_intermediate_result(sg_program_list, sg_program_idx, nmn_program_list, nmn_more_data, video_id, answer):
    def post_process_sg_res_by_step(sg_res_by_step):
        # delete some intermediate results in sg_res_by_step that we can't handle now
        keys_to_delete = list()
        for key in sg_res_by_step:
            if callable(sg_res_by_step[key]):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del sg_res_by_step[key]
        return sg_res_by_step

    if sg_executer is None:
        raise ValueError('sg_executer not initialized!')

    nmn_program_idx, nmn_existsframe_filterframe_idx_mapping = nmn_more_data['idx_list'], nmn_more_data['existsframe_filterframe_idx_mapping']
    frame_idxs = [idx for prog, idx in zip(nmn_program_list, nmn_program_idx) if isinstance(prog, str) and 'Frame' in prog]

    try:
        sg_res, sg_res_by_step, video_metadata = sg_executer(
            program_list=sg_program_list, program_idxs=sg_program_idx, video_id=video_id, frame_idxs=frame_idxs,
            frame_idx_mapping=nmn_existsframe_filterframe_idx_mapping
        )
        sg_res_by_step = post_process_sg_res_by_step(sg_res_by_step)
        if sg_res != answer:
            return None
    except Exception:
        return None

    return sg_res_by_step


def get_program_list_string_index(program_list, question):
    '''
    This function is used to link strings in program to word / phrases in question.
    In STAIR, we get the features of these strings with question encoders in the context of the question, and substitiute the strings in program with these features as NMN input.

    returns: span by word index and span by char index
    '''
    if program_list is None:
        return None, None

    def get_start_idx_of_sub_sequence(big_list, small_list):
        '''
        as items in small_list do not often occur in big_list, we use simple matching instead of KMP algorithm here.
        '''
        for i in range(len(big_list) - len(small_list)):
            if big_list[i: i + len(small_list)] == small_list:
                return i
        return None

    question_words = word_tokenize(question)
    question_token_start_end_idxs = [(0, 0)]
    for word in question_words:
        start_index = question.index(word, question_token_start_end_idxs[-1][0])
        end_index = start_index + len(word)
        question_token_start_end_idxs.append((start_index, end_index))
    del question_token_start_end_idxs[0]

    question_words = [rules_dict_question.get(w, w) for w in question_words]
    word_and_pos_list = nltk.pos_tag(question_words)
    word_and_pos_list = [(w, 'V') if w.endswith('ing') else (w, pos) for w, pos in word_and_pos_list]
    question_words = [wnl.lemmatize(w, p[0].lower()) if p[0].lower() in ['v', 'n'] and w != 'clothes' else w for w, p in word_and_pos_list]

    span_by_word_index, span_by_char_index = dict(), dict()
    for i, prog in enumerate(program_list):
        if prog in ALL_KWS:
            continue
        prog = prog.replace('_', ' ')
        prog = rules_dict_prog.get(prog, prog)
        prog_word_list = [rules_dict_prog.get(w, w) for w in word_tokenize(prog)]
        word_and_pos_list = nltk.pos_tag(prog_word_list)
        verb = None
        new_prog_word_list = list()
        for prog_word, pos in word_and_pos_list:
            if pos[0] in ['V', 'N']:
                verb = wnl.lemmatize(prog_word, pos[0].lower())
                new_prog_word_list.append(verb)
            else:
                new_prog_word_list.append(prog_word)

        start_idx = get_start_idx_of_sub_sequence(question_words, new_prog_word_list)
        if start_idx is None:
            span_by_word_index[i] = (None, None)
            span_by_char_index[i] = (None, None)
        else:
            end_idx = start_idx + len(new_prog_word_list)
            span_by_word_index[i] = (start_idx, end_idx)
            span_by_char_index[i] = question_token_start_end_idxs[start_idx][0], question_token_start_end_idxs[end_idx-1][1]
    return span_by_word_index, span_by_char_index


def convert_(example):
    new_dict = {
        key: example[key] for key in ['question', 'answer', 'video_id', 'program', 'qa_id', 'novel_comp', 'more_steps']
    }
    new_dict['nmn_program'], nmn_more_data = program_parser.parse_program(example['program'])
    new_dict['nmn_program_idx'] = nmn_more_data['idx_list']
    new_dict['sg_program'], new_dict['sg_program_idx'] = scene_graphs.parse_program(example['program'])
    
    # get sg intermediate result
    new_dict['sg_res_by_step'] = get_scene_graph_intermediate_result(
        sg_program_list=new_dict['sg_program'], sg_program_idx=new_dict['sg_program_idx'],
        nmn_program_list=new_dict['nmn_program'], nmn_more_data=nmn_more_data,
        video_id=new_dict['video_id'], answer=new_dict['answer']
    )

    # get index spans of the question for strings in nmn_program
    span_by_word_index, span_by_char_index = get_program_list_string_index(
        program_list=new_dict['nmn_program'], question=new_dict['question']
    )
    new_dict['nmn_program_span_by_word'] = span_by_word_index
    new_dict['nmn_program_span_by_char'] = span_by_char_index
    return new_dict


def load_generated_program_fairseq(generated_program_filename):
    print('loading generated data from file:', generated_program_filename)
    generated_programs = dict()

    with open(generated_program_filename) as f_in:
        question_index = None
        for i, line in enumerate(f_in):
            if line.startswith('S'):       # starts a new program
                if question_index is not None:      # program parser cant generate a valid program for the last question
                    generated_programs[question_index] = None
                question_index = int(line.split('\t')[0][2:])
            elif line.startswith('D'):
                if question_index is not None:
                    program_str = line.strip().split('\t')[-1]
                    program_list = program_str.split(' ')[::-1]
                    if program_parser.program_is_valid(program_list):
                        generated_programs[question_index] = program_list
                        question_index = None

    print('loaded %d programs' % len(generated_programs))
    return generated_programs


def load_generated_program_huggingface(generated_program_filename):
    generated_programs = dict()
    for line in open(generated_program_filename).readlines():
        try:
            question_id, question, program = line.strip().split('\t')
        except:
            continue        # sometimes the generated program or question may contain \n ??

        if question_id in generated_programs:
            continue
        program_list = program.split(' ')
        # make some simple corrections
        program_list = [
            'while' if p in ['when', 'with'] else
            'video' if p.lower() in ['next'] else
            p for p in program_list
        ]
        if program_parser.program_is_valid(program_list):
            generated_programs[question_id] = program_list
    return generated_programs


def upgrade_pkl_with_generated_program(src_pkl_filename, generated_program_filename, dest_pkl_filename, generated_format='fairseq'):
    '''
    generated_program: fairseq format result file
    '''
    if generated_format == 'fairseq':
        load_generated_program = load_generated_program_fairseq
    elif generated_format == 'huggingface':
        load_generated_program = load_generated_program_huggingface
    else:
        raise ValueError('generated_format must be one of: fairseq, huggingface')
    generated_programs = load_generated_program(generated_program_filename)

    # load generated_program_filename
    num_valid_programs = sum([val is not None for val in generated_programs.values()])
    print('number of valid programs:', num_valid_programs)
    print('some of the keys:', random.sample(generated_programs.keys(), 50))

    src_pkl = pickle.load(open(src_pkl_filename, 'rb'))
    dest_pkl = list()
    num_diff_programs = 0
    for i, example in enumerate(src_pkl):
        new_example = {key: example[key] for key in ['question', 'answer', 'video_id', 'program', 'qa_id']}
        nmn_program = generated_programs.get(new_example['qa_id'], None)
        if nmn_program == example['nmn_program']:       # if the coming nmn_program is identical to the original program, we do not need to do anything again
            for key in ['nmn_program', 'nmn_program_span_by_word', 'nmn_program_span_by_char']:
                new_example[key] = example[key]
        else:
            num_diff_programs += 1
            new_example['nmn_program'] = nmn_program
            span_by_word_index, span_by_char_index = get_program_list_string_index(
                program_list=new_example['nmn_program'], question=new_example['question']
            )
            new_example['nmn_program_span_by_word'] = span_by_word_index
            new_example['nmn_program_span_by_char'] = span_by_char_index
        dest_pkl.append(new_example)

    print('number of different programs:', num_diff_programs)

    with open(dest_pkl_filename, 'wb') as f_out:
        pickle.dump(dest_pkl, f_out)


def merge_json_data_program(src_data_filename, generated_program_filename, dest_data_filename, dataset='STAR', generated_format='huggingface'):
    def load_data(filename, dataset='STAR'):
        if dataset in ['STAR', 'MSRVTT']:
            return json.load(open(filename))
        elif dataset == 'NEXTQA':
            df = pd.read_csv(filename)
            ret = list()
            for idx, line in df.iterrows():
                ret.append({
                    'video_id': str(line['video']), 'question': line['question'], 'answer': line['answer'],
                    'question_id': str(idx), 'choices': [{'choice': line['a%d' % i]} for i in range(5)]
                })
            return ret

    if generated_format == 'fairseq':
        load_generated_program = load_generated_program_fairseq
    elif generated_format == 'huggingface':
        load_generated_program = load_generated_program_huggingface
    else:
        raise ValueError('generated_format must be one of: fairseq, huggingface')
    generated_programs = load_generated_program(generated_program_filename)
    print('loaded %d generated_programs' % len(generated_programs))
    src_data = load_data(src_data_filename, dataset)
    print('loaded %d src_data' % len(src_data))

    dest_data = list()
    no_program_num, no_span_num, total_spam_num = 0, 0, 0
    wanted_keys = {
        'STAR': ['question_id', 'question', 'answer', 'choices', 'video_id', 'start', 'end'],
        'MSRVTT': ['question_id', 'question', 'answer', 'video', 'answer_type'],
        'NEXTQA': ['question_id', 'question', 'answer', 'choices', 'video_id'],
    }
    for example in src_data:
        new_example = {key: example[key] for key in wanted_keys[dataset] if key in example}
        nmn_program = generated_programs.get(new_example['question_id'], None)
        if dataset == 'STAR':
            new_example['question'] = new_example['question'].replace('/', ' ')
            new_example['choices'] = [{'choice_id': c['choice_id'], 'choice': c['choice'].replace('/', ' ')} for c in new_example['choices']]
            if 'answer' in new_example:
                new_example['answer'] = new_example['answer'].replace('/', ' ')

        if nmn_program is None:
            no_program_num += 1
            new_example['nmn_program'] = []
            new_example['nmn_program_span_by_word'] = None
            new_example['nmn_program_span_by_char'] = None
        else:
            new_example['nmn_program'] = nmn_program
            span_by_word_index, span_by_char_index = get_program_list_string_index(
                program_list=new_example['nmn_program'], question=new_example['question']
            )
            new_example['nmn_program_span_by_word'] = span_by_word_index
            new_example['nmn_program_span_by_char'] = span_by_char_index
            total_spam_num += len(span_by_word_index)
            no_span_num += len([v for v in span_by_word_index.values() if None in v])
        dest_data.append(new_example)
    print('examples with no programs: %d' % no_program_num)

    with open(dest_data_filename, 'wb') as f_out:
        pickle.dump(dest_data, f_out)

    print('total_example_num: %d' % len(dest_data))
    print('no_program_num: %d' % no_program_num)
    print('total_spam_num: %d' % total_spam_num)
    print('no_span_num: %d' % no_span_num)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, choices=['convert', 'upgrade'])

    # convert
    parser.add_argument('--train-sg-filename', type=str, default=None)
    parser.add_argument('--test-sg-filename', type=str, default=None)
    parser.add_argument('--id2word-filename', type=str)
    parser.add_argument('--word2id-filename', type=str)
    parser.add_argument('--num-workers', type=int, default=20)

    parser.add_argument('--train-csv-filename', type=str, default=None)
    parser.add_argument('--test-csv-filename', type=str, default=None)
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-folder', type=str)

    # upgrade
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--generated-format', type=str, default='fairseq')
    parser.add_argument('--src-data-filename', type=str)
    parser.add_argument('--dest-data-filename', type=str)
    parser.add_argument('--generated-filename', type=str)

    args = parser.parse_args()
    print('args:', args)

    if args.func == 'convert':
        os.makedirs(args.output_folder, exist_ok=True)
        sg_filenames = list()
        if args.train_sg_filename:
            sg_filenames.append(args.train_sg_filename)
        if args.test_sg_filename:
            sg_filenames.append(args.test_sg_filename)

        # 1. init scene graph executer
        sg_executer = scene_graphs.SceneGraphExecuter(
            sg=sg_filenames,
            id2word_filename=args.id2word_filename,
            word2id_filename=args.word2id_filename
        )

        # 2. split train set into my train and valid set. use 10% examples as valid set
        train_filename = os.path.join(args.input_folder, 'train_balanced.txt')
        train_valid_data = json.load(open(train_filename))

        if args.train_csv_filename is not None:
            csv_data = pd.read_csv(args.train_csv_filename, sep=',')
            qa_ids = list(csv_data['key'])
        else:
            qa_ids = list(train_valid_data.keys())
        train_length = int(len(qa_ids) * 0.9)
        train_qaids = qa_ids[:train_length]
        valid_qaids = qa_ids[train_length:]

        train_examples, valid_examples = list(), list()
        for i, qa_id in enumerate(train_qaids):
            data = train_valid_data[qa_id]
            data['qa_id'] = qa_id
            train_examples.append(data)

        for i, qa_id in enumerate(valid_qaids):
            data = train_valid_data[qa_id]
            data['qa_id'] = qa_id
            valid_examples.append(data)

        # multithreading
        print('converting valid examples using multithreading, num workers =', args.num_workers)
        tic = datetime.datetime.now()
        with Pool(args.num_workers) as p:
            valid_data_pkl = p.map(convert_, valid_examples)
        toc = datetime.datetime.now()
        print('converted %d valid examples' % len(valid_examples))
        print('time used:', toc-tic)
        pickle.dump(valid_data_pkl, open(os.path.join(args.output_folder, 'valid_balanced.pkl'), 'wb'))

        print('converting train examples using multithreading, num workers =', args.num_workers)
        tic = datetime.datetime.now()
        with Pool(args.num_workers) as p:
            train_data_pkl = p.map(convert_, train_examples)
        toc = datetime.datetime.now()
        print('converted %d train examples' % len(train_examples))
        print('time used:', toc-tic)
        pickle.dump(train_data_pkl, open(os.path.join(args.output_folder, 'train_balanced.pkl'), 'wb'))

        test_filename = os.path.join(args.input_folder, 'test_balanced.txt')
        test_data = json.load(open(test_filename))

        if args.test_csv_filename is not None:
            # sort examples in test set according to qa_id in csv file, for easier analyzing model output
            csv_data = pd.read_csv(args.test_csv_filename, sep=',')
            qa_ids = list(csv_data['key'])
        else:
            qa_ids = list(test_data.keys())

        test_examples = list()
        for i, qa_id in enumerate(qa_ids):
            data = test_data[qa_id]
            data['qa_id'] = qa_id
            test_examples.append(data)

        print('converting test examples using multithreading, num workers =', args.num_workers)
        tic = datetime.datetime.now()
        with Pool(args.num_workers) as p:
            test_data_pkl = p.map(convert_, test_examples)
        toc = datetime.datetime.now()
        print('converted %d test examples' % len(test_examples))
        print('time used:', toc-tic)
        pickle.dump(test_data_pkl, open(os.path.join(args.output_folder, 'test_balanced.pkl'), 'wb'))

    elif args.func == 'upgrade':
        if args.dataset == 'AGQA':
            upgrade_pkl_with_generated_program(args.src_data_filename, args.generated_filename, args.dest_data_filename, args.generated_format)
        elif args.dataset in ['STAR', 'MSRVTT', 'NEXTQA']:
            merge_json_data_program(args.src_data_filename, args.generated_filename, args.dest_data_filename, args.dataset, args.generated_format)
        else:
            raise NotImplementedError()


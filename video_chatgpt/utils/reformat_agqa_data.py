# merge filter outputs of STAIR into AGQA questions
import sys
import os
import json
import pickle
import random

from tqdm import tqdm

import argparse


def load_filter_data(filter_fname):
    i = 0
    ret = dict()
    if '%d' in filter_fname:
        while os.path.isfile(filter_fname % i):
            ret = {**ret, **pickle.load(open(filter_fname % i, 'rb'))}
            i += 1
    else:
        ret = pickle.load(open(filter_fname, 'rb'))
    print('loaded %d filter data examples' % len(ret))
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_ratio', type=float, default=0.01)
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--filter_fname', type=str)
    parser.add_argument('--output_fname', type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    src_data = json.load(open(args.input_fname))
    sample_num = int(len(src_data) * args.sample_ratio)
    sampled_qids = random.sample(src_data.keys(), sample_num)

    if args.filter_fname is not None:
        print('adding STAIR filter output')
        filter_data = load_filter_data(args.filter_fname)

    new_data = list()
    for qid in tqdm(sampled_qids):
        example = src_data[qid]
        if args.filter_fname is not None:
            filter_output_list = list(filter_data.get(qid, dict()).values())
            filter_output_list.sort(key=lambda x: -x[0])        # from lower level to higher level
            filter_output_texts = list()
            for _, kw, ans_list in filter_output_list:
                # 1 answer per filter module, max 3 filter modules
                for ans in ans_list[:1]:
                    filter_output_texts.append('%s %s.' % (kw, ans))
                filter_output_texts = filter_output_texts[:3]
        else:
            filter_output_texts = []

        new_data.append({
            'question': 'Possible useful information in video: %s Question: %s' % (
                ' '.join(filter_output_texts), example['question']) if len(filter_output_texts) else example['question'],
            'answer': example['answer'], 'question_id': qid, 'video_name': example['video_id']
        })

    print('writing %d examples to %s' % (len(new_data), args.output_fname))
    with open(args.output_fname, 'w') as f_out:
        json.dump(new_data, f_out)

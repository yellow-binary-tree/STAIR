import os
import json
import math
import pickle
import datetime
import traceback
import random
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoTokenizer

from video_nmn.args import get_args
from video_nmn.dataset import AGQADataset, STARDataset, MSRVTTDataset, to_device, collate_fn
from video_nmn.module_net import VideoNMN, L2Normalize
from utils.program_parser import stat_module_levels, get_childrens_and_parents


def star_format_test_output(preds_golds_list):
    res = {key: list() for key in ['Interaction', 'Sequence', 'Prediction', 'Feasibility']}
    for qa_id, pred in zip(preds_golds_list['qa_ids'], preds_golds_list['preds']):
        res[qa_id.split('_')[0]].append({'question_id': qa_id, 'answer': pred})
    return res


@torch.no_grad()
def evaluate(dataloader, model, criterion, preds_file=None):
    acc_list, preds_golds_list = list(), {'preds': [], 'golds': [], 'qa_ids': []}
    errors = 0

    for i, batch in enumerate(dataloader):
        if i % 1000 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time, 'test progress: %d batches, error examples: %d' % (i, errors))
        batch = to_device(batch, device)
        model_output = model(batch, return_res_by_step=False, test_mode=True)
        logits = model_output['logits']

        pred_list = torch.argmax(logits, dim=1)
        if args.dataset in ['AGQA', 'MSRVTT']:
            for pred, gold, qa_id in zip(pred_list, batch['answer'], batch['qa_id']):
                preds_golds_list['qa_ids'].append(qa_id)
                unk_token_id = dataloader.dataset.answer_vocab['word2id']['<UNK>']
                acc_list.append(pred.cpu().item() == gold.cpu().item() and gold.cpu().item() != unk_token_id)
                preds_golds_list['preds'].append(dataloader.dataset.answer_vocab['id2word'][pred.cpu().item()])
                preds_golds_list['golds'].append(dataloader.dataset.answer_vocab['id2word'][gold.cpu().item()])
        elif args.dataset == 'STAR':
            for pred, qa_id in zip(pred_list, batch['qa_id']):
                preds_golds_list['preds'].append(pred.cpu().item())
                preds_golds_list['qa_ids'].append(qa_id)

    if preds_file is not None:
        if args.dataset == 'STAR':
            preds_golds_list = star_format_test_output(preds_golds_list)
        json.dump(preds_golds_list, open(preds_file, 'w'))
    
    if args.dataset in ['AGQA', 'MSRVTT']:
        return sum(acc_list) / len(acc_list), 0.    # , sum(loss_list) / len(loss_list)
    elif args.dataset == 'STAR':        # STAR uses website online evaluation for test set performance 
        return 0, 0.


def get_filter_text_results(dataloader, model, filter_vocab_filename=None, result_filename=None):
    def get_kw_representations(filter_vocab):
        filter_ans_reps = list()
        for answer in filter_vocab:
            answer_embedding = dataloader.dataset.embed_sent(answer)  # .unsqueeze(0)
            # answer_embedding = pack_sequence(answer_embedding)
            _, answer_rep = model.encode_question_no_grad(answer_embedding.to(device))
            answer_rep = model.contrastive_head(answer_rep.squeeze())
            filter_ans_reps.append(answer_rep)
        filter_ans_reps = torch.stack(filter_ans_reps)
        return filter_ans_reps

    # get representations of all string in the vocab
    filter_vocab = json.load(open(filter_vocab_filename))
    filter_ans_reps = get_kw_representations(filter_vocab)
    filter_results_text_list = dict()

    progress = -1
    for batch_i, batch in enumerate(dataloader):        # batch size = 1 for NMN
        batch = to_device(batch, device)
        model_output = model(batch, return_res_by_step=False, return_result_of_each_step=True, test_mode=True)

        qa_id, nmn_program, result_of_each_step, nmn_program_idx = batch['qa_id'], batch['nmn_program_list'], model_output['result_of_each_step'], batch['nmn_program_idx']
        progress += 1
        if progress % 5000 == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'progress: %d / %d' % (progress, len(dataloader.dataset)))

        if nmn_program is None:
            continue

        if nmn_program_idx is None:
            nmn_program_idx = list(range(len(nmn_program)))

        try:
            nmn_program_levels = stat_module_levels(nmn_program)
            childrens_list, parents_list = get_childrens_and_parents(nmn_program)
            filter_results_text = dict()

            for i, (prog, prog_idx, prog_level, childrens) in enumerate(zip(nmn_program, nmn_program_idx, nmn_program_levels, childrens_list)):
                if prog != 'Filter':
                    continue
                params, result = result_of_each_step[i]
                gold_sims = nn.CosineSimilarity()(result.unsqueeze(0), filter_ans_reps)
                gold_ranks = torch.argsort(gold_sims, descending=True)
                preds_top_10 = [filter_vocab[i] for i in gold_ranks[:10]]
                filter_results_text[prog_idx] = (prog_level, nmn_program[childrens[1]].replace('_', ' '), preds_top_10)
            filter_results_text_list[qa_id] = filter_results_text      # level, keyword, results
        except:
            print('exception when getting filer results, skipping', qa_id, 'Traceback:')
            print(traceback.format_exc())

    with open(result_filename, 'wb') as f_out:
        pickle.dump(filter_results_text_list, f_out)


if __name__ == '__main__':
    print('EVALUATE TIME:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    args = get_args()
    if args.result_filename is None:
        args.result_filename = 'result.json'
    # args.start_index = 0
    # args.end_index = -1
    print(args)

    # load config and model
    model_config, model = None, None
    if args.model_ckpt is not None:
        # if os.path.isdir(args.model_ckpt):
        #     model_config = json.load(open(os.path.join(args.model_ckpt, 'config.json')))
        #     model = torch.load(os.path.join(args.model_ckpt, 'pytorch_model.bin'))
        # else:
        #     model = torch.load(args.model_ckpt)
        model_config = json.load(open(os.path.join(args.model_ckpt, 'config.json')))
        model = VideoNMN(model_config)
        model.load_state_dict(torch.load(os.path.join(args.model_ckpt, 'pytorch_model.bin')))

    print('MODEL STRUCTURE')
    print(model)

    model.eval()
    model.to(device)

    if args.evaluate_func == 'acc':
        if args.dataset == 'AGQA':
            test_dataset = AGQADataset(args, split='test')
        elif args.dataset == 'STAR':
            test_dataset = STARDataset(args, split='test')
        elif args.dataset == 'MSRVTT':
            test_dataset = MSRVTTDataset(args, split='test')
        else:
            raise NotImplementedError

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        criterion = nn.CrossEntropyLoss()
        acc, loss = evaluate(test_dataloader, model, criterion, os.path.join(args.output, args.result_filename))
        print('valid acc: %.4lf; valid loss: %.4lf' % (acc, loss))

    elif args.evaluate_func == 'filter_text_result':
        os.makedirs(os.path.dirname(args.result_filename), exist_ok=True)
        print('function filter_text_result')
        test_dataset = AGQADataset(args, split='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        get_filter_text_results(test_dataloader, model, args.filter_answer_vocab_filename, args.result_filename)

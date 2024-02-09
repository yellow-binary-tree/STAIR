import os
import shutil
import json
import numpy as np
import torch
import math
import pickle
from torch import nn
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from video_nmn.args import get_args
from video_nmn.module_net import VideoNMN
from video_nmn.dataset import AGQADataset, collate_fn, to_device

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def backup_code(output_folder):
    os.makedirs(os.path.join(output_folder, 'code'), exist_ok=True)
    if os.path.exists(os.path.join(output_folder, 'code', 'video_nmn')):
        shutil.rmtree(os.path.join(output_folder, 'code', 'video_nmn'))
    shutil.copytree('./video_nmn', os.path.join(output_folder, 'code', 'video_nmn'))
    if os.path.exists(os.path.join(output_folder, 'code', 'utils')):
        shutil.rmtree(os.path.join(output_folder, 'code', 'utils'))
    shutil.copytree('./utils', os.path.join(output_folder, 'code', 'utils'))
    shutil.copy('./train_module.py', os.path.join(output_folder, 'code'))


class CriterionByModule:
    def __init__(self, args):
        self.module_loss_type = 'cont_nolinear'
        self.criterions = {
            'Exists': self.criterion_exists,
            'Xor': self.criterion_exists,
            'Equals': self.criterion_equals,
            'Filter': self.criterion_filter,
            'ToAction': self.criterion_toaction,
            'FilterFrame': self.criterion_filterframe,
            'ExistsFrame': self.criterion_existsframe,
            'Superlative': self.criterion_superlative,
            'Localize': self.criterion_localize,
            'Temporal': self.criterion_temporal,
            'decoder': self.criterion_decoder,
        }
        # some modules do not have parameters, so we don't need to pretrain them
        word2id = json.load(open(args.word2id_filename))        # some text in this file maps to the same id
        id_list = sorted(list(set(word2id.values())))      # this is the real vocab_size
        self.id2index = {id_: i for i, id_ in enumerate(id_list)}
        self.word2id = dict()
        for word, id_ in word2id.items():
            self.word2id[word] = self.id2index[id_]

        print('criterion word2id length', len(self.word2id))
        print('criterion vocab size', len(self.id2index))

    def __call__(self, module_name, pred, gold):
        '''
        pred: NMN module output
        gold: scene graph executer output
        '''
        return self.criterions[module_name](pred, gold)

    def span_to_attention(self, gold, video_length):
        '''
        used in attention score criterions, i.e., Localize, Temporal, FilterFrame
        '''
        gold_tensor = torch.zeros(video_length).to(device)
        start, end = min(video_length - 0.002, max(0.001, gold[0])), min(video_length - 0.001, gold[1])
        start_int, end_int = math.ceil(start), math.floor(end)
        if start_int < end_int:
            gold_tensor[start_int:end_int] += 1
        if start_int <= end_int:
            gold_tensor[start_int - 1] += start_int - start
            gold_tensor[end_int] += end - end_int
        else:
            gold_tensor[end_int] += end - start
        return gold_tensor

    def attention_score_criterion(self, pred, gold):
        '''
        this works for (pred, gold) of shape both [num_frames] and [batch_size, num_frames]
        '''
        gold_tensor = torch.stack([gold, 1-gold], dim=-1)
        pred_tensor = torch.stack([pred, 1-pred], dim=-1)
        loss_by_time = -torch.sum(torch.log(pred_tensor) * gold_tensor, dim=-1)
        return torch.mean(loss_by_time)

    def criterion_exists(self, pred, gold):
        '''
        pred: tensor [2]
        gold: True or False
        '''
        pred_tensor = pred.unsqueeze(0)
        gold_tensor = torch.tensor([gold], device=pred_tensor.device, dtype=torch.long)
        return nn.CrossEntropyLoss()(pred_tensor, gold_tensor)

    def criterion_equals(self, pred, gold):
        '''
        pred: scalar tensor, similarity of the two inputs
        gold: bool
        '''
        gold_tensor = torch.tensor(gold, device=pred.device, dtype=torch.long)
        return torch.mean(torch.square(pred - gold_tensor))

    def criterion_compare(self, pred, gold):
        gold = False if gold == 'before' else True
        return self.criterion_exists(pred, gold)

    def criterion_filter(self, pred, gold):
        '''
        if mse-module-loss:
            pred, gold: tensor, [hidden_size]
        else:
            pred: tensor, [obj_class_num]
            gold: str, or list of str, object class name
        
        '''
        if self.module_loss_type in ['cont', 'cont_nolinear']:
            logits = torch.matmul(pred, gold.t()).unsqueeze(0)
            label = torch.tensor([0], device=logits.device)
            return nn.CrossEntropyLoss()(logits, label)

        elif self.module_loss_type == 'cont-valid':
            if gold == []:      # no results found
                return torch.tensor(0., device=pred.device)
            else:
                gold_tensor = torch.stack([g[1] for g in gold]).mean(dim=0)
                return nn.CosineSimilarity(dim=0)(pred, gold_tensor)

    def criterion_toaction(self, pred, gold):
        '''
        pred: tensor, [obj_class_num]
        gold: str, or list of str, object class name
        '''
        return self.criterion_filter(pred, gold)

    def criterion_filterframe(self, pred, gold):
        '''
        pred: tensor, [num_frames, obj_class_num]
        gold: dict {entity_name: FrameInterval}
        '''
        num_frames = pred.size(0)
        gold_tensor = torch.zeros_like(pred)
        for key, val in gold.items():
            key_id = self.word2id[key]
            gold_tensor_for_this_id = self.span_to_attention(val, num_frames)
            gold_tensor[:, key_id] = gold_tensor_for_this_id
        gold_tensor_sum = torch.sum(gold_tensor, dim=1, keepdim=True)
        gold_tensor = gold_tensor / gold_tensor_sum
        gold_tensor = torch.where(gold_tensor.isinf() | gold_tensor.isnan(), torch.zeros_like(gold_tensor), gold_tensor)
        return nn.BCELoss()(nn.Softmax(dim=1)(pred), gold_tensor)

    def criterion_existsframe(self, pred, gold):
        '''
        pred: attention tensor, [num_frames]
        gold: a FrameInterval object
        '''
        num_frames = pred.size(0)
        gold_tensor = self.span_to_attention(gold, num_frames)
        return self.attention_score_criterion(pred, gold_tensor)

    def criterion_superlative(self, pred, gold):
        '''
        pred: tensor, [text_size]
        gold: tensor, [text_size]
        '''
        return self.criterion_filter(pred, gold)

    def criterion_localize(self, pred, gold):
        '''
        pred: tensor, [num_kws, num_frames]
        gold: a list of FrameInterval objects
        '''
        num_frames = pred.size(1)
        gold_tensor = torch.zeros_like(pred)
        for i in range(pred.size(0)):
            gold_tensor[i] = self.span_to_attention(gold[i], num_frames)
        return self.attention_score_criterion(pred, gold_tensor)

    def criterion_temporal(self, pred, gold):
        '''
        pred: tensor, [num_frames]
        gold: a FrameInterval object
        '''
        num_frames = pred.size(0)
        gold_tensor = self.span_to_attention(gold, num_frames)
        return self.attention_score_criterion(pred, gold_tensor)

    def criterion_decoder(self, pred, gold):
        return nn.CrossEntropyLoss()(pred.unsqueeze(0), gold.unsqueeze(0))


def post_process_sg_res_by_step(sg_res_by_step):
    # delete some intermediate results in sg_res_by_step that we can't handle now
    keys_to_delete = list()
    for key in sg_res_by_step:
        # convert strings into embeddings, but not the ones produced by Query
        if isinstance(sg_res_by_step[key], list):
            if not isinstance(sg_res_by_step[key][0], str):
                keys_to_delete.append(key)
        if callable(sg_res_by_step[key]):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del sg_res_by_step[key]
    return sg_res_by_step


def save_model(output_dir, model, config, args):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model, os.path.join(output_dir, 'pytorch_model.bin'))
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'))
    pickle.dump(args, open(os.path.join(output_dir, 'args.pkl'), 'wb'))


@torch.no_grad()
def evaluate_by_module(args, dataloader, model, criterions, preds_file=None):
    losses = {module: [] for module in criterions.criterions.keys()}
    acc_list, preds_golds_list = list(), {'preds': [], 'golds': [], 'qa_ids': []}
    unk_token_id = dataloader.dataset.answer_vocab['word2id']['<UNK>']

    criterions_module_loss_type = criterions.module_loss_type
    if criterions_module_loss_type in ['cont', 'cont_nolinear']:
        criterions.module_loss_type = 'cont-valid'        # set loss type to mse during evaluation to make loss invarient to other examples in batch

    for batch in dataloader:
        batch = to_device(batch, device)

        model_output = model(batch, return_res_by_step=args.module_loss_weight != 0)
        answer, logits, res_by_step = batch['answer'], model_output['logits'], model_output['res_by_step']
        sg_res_by_step = model_output['sg_res_by_step']

        for step, (module, nmn_res) in res_by_step.items():
            if step not in sg_res_by_step or module not in criterions.criterions:
                continue
            sg_res = sg_res_by_step[step]
            if sg_res is None:
                continue
            loss = criterions(module, nmn_res, sg_res)
            losses[module].append(loss.detach().cpu().item())
            loss = loss * args.module_loss_weight / args.gradient_accumulation

        # final loss for training the classifier
        loss = criterions('decoder', logits, answer)
        losses['decoder'].append(loss.detach().cpu().item())
        loss = loss / args.gradient_accumulation

        # calculate accuracy on validation set
        pred = torch.argmax(logits)
        acc_list.append(answer.cpu().item() == pred.cpu().item() and answer.cpu().item() != unk_token_id)
        preds_golds_list['preds'].append(dataloader.dataset.answer_vocab['id2word'][pred.cpu().item()])
        preds_golds_list['golds'].append(dataloader.dataset.answer_vocab['id2word'][answer.cpu().item()])
        preds_golds_list['qa_ids'].append(batch['qa_id'])

    print('evaluation finished, evaluated on %d examples' % len(acc_list))
    criterions.module_loss_type = criterions_module_loss_type       # restore criterions.module_loss_type back
    valid_losses = dict()
    for module, ls in losses.items():
        if len(ls) == 0:
            valid_losses[module] = float('inf')
        else:
            valid_losses[module] = sum(ls) / len(ls)

    if preds_file is not None:
        json.dump(preds_golds_list, open(preds_file, 'w'))

    return sum(acc_list) / len(acc_list), valid_losses


def main(args):
    # load dataset
    if args.dataset == 'AGQA':
        train_dataset = AGQADataset(args, split='train')
        if args.debug:
            valid_dataset = train_dataset
            print('using train_dataset as valid_dataset in debug mode')
        else:
            valid_dataset = AGQADataset(args, split='valid')
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        raise NotImplementedError

    # create criterions
    criterions = CriterionByModule(args)

    # load config and model
    model_config, model = None, None
    if args.model_ckpt is not None:
        print('loading model from', args.model_ckpt)
        if os.path.isdir(args.model_ckpt):
            model_config = json.load(open(os.path.join(args.model_ckpt, 'config.json')))
            model = torch.load(os.path.join(args.model_ckpt, 'pytorch_model.bin'))
        else:
            model = torch.load(args.model_ckpt)

    if model_config is None:
        if args.config_filename is not None:
            model_config = json.load(open(args.config_filename))
        else:
            model_config = {
                'hidden_size': args.hidden_size, 'video_size': args.video_size, 'text_size': args.text_size,
                'dropout': args.dropout, 'answer_vocab_length': train_dataset.answer_vocab_length(),
                'max_video_length': args.max_video_length, 'init_method': args.init_method,
                'layer_norm': args.layer_norm,
                'have_pretrain_head': args.module_loss_weight != 0, 'object_types': len(criterions.id2index),
            }

    print('model config:', model_config)
    if model is None:
        model = VideoNMN(model_config, debug=args.debug, pretrain_modules=set(criterions.criterions.keys()))
    model.train()
    model.to(device)

    # load tensorboard
    tensorboard_dir = os.path.join(args.output, 'runs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tb_writer = SummaryWriter(tensorboard_dir)

    # losses and optimizers
    losses = {module: [] for module in criterions.criterions.keys()}

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    def lr_lambda(iter_):
        if iter_ > args.scheduler_total_iters:
            return args.scheduler_end_factor
        return args.scheduler_start_factor + (args.scheduler_end_factor - args.scheduler_start_factor) / args.scheduler_total_iters * iter_
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # main training loop
    global_steps, skipped_num, best_acc = 0, 0, 0.

    class_reps, neg_reps = dict(),  dict()
    module_called_times_list = list()
    batch_loss = 0.

    for epoch_i in range(args.num_epochs):
        for batch in train_dataloader:
            global_steps += 1
            batch = to_device(batch, device)
            model_output = model(batch, return_res_by_step=args.module_loss_weight != 0)
            answer, logits, res_by_step = batch['answer'], model_output['logits'], model_output['res_by_step']
            sg_res_by_step = model_output['sg_res_by_step']

            example_loss = 0.
            if global_steps < args.train_module_before_iters:
                for step, (module, nmn_res) in res_by_step.items():
                    # if args.debug:
                    #     print('cutting down into module:', step, module)
                    if step not in sg_res_by_step or module in args.modules_no_intermediate_train or module not in criterions.criterions:
                        continue
                    sg_res = sg_res_by_step[step]
                    if sg_res is None:      # sometimes sg_executer can't get correct results
                        continue

                    if module in ['Filter', 'Superlative', 'ToAction']:
                        # record intermediate results here, and calculate loss after the batch finishes
                        for class_name, class_rep in sg_res:
                            if class_name not in class_reps:
                                class_reps[class_name] = list()
                            class_reps[class_name].append(((global_steps - 1) % args.gradient_accumulation, module, nmn_res))
                            neg_reps[class_name] = class_rep        # all representations of the same class_name in neg_reps should be the same
                    else:
                        loss = criterions(module, nmn_res, sg_res)
                        losses[module].append(loss.detach().cpu().item())
                        if len(losses[module]) > args.report_interval:
                            del losses[module][0]
                        loss = loss * args.module_loss_weight / args.gradient_accumulation
                        example_loss += loss

            # final loss for training the classifier
            if global_steps > args.train_decoder_after_iters:
                loss = criterions('decoder', logits, answer)
                losses['decoder'].append(loss.detach().cpu().item())
                loss = loss * args.decoder_loss_weight / args.gradient_accumulation
                example_loss += loss

            batch_loss += example_loss
            if len(losses['decoder']) > args.report_interval:
                del losses['decoder'][0]

            if global_steps % args.gradient_accumulation == 0:
                if global_steps < args.train_module_before_iters:
                    cont_loss = 0.
                    for class_name, class_values in class_reps.items():
                        for iter_no, module, nmn_res in class_values:
                            pos = neg_reps[class_name]
                            neg = [val for key, val in neg_reps.items() if key != class_name]       # hidden reps from words
                            if neg:
                                neg = torch.stack(neg)
                                gold = torch.cat([pos.unsqueeze(0), neg])       # [num_examples, hidden_size], where the first row is posisive example
                            else:
                                gold = pos.unsqueeze(0)
                            # print(module, class_name, nmn_res.size(), gold.size())
                            loss = criterions(module, nmn_res, gold)
                            losses[module].append(loss.detach().cpu().item())
                            if len(losses[module]) > args.report_interval:
                                del losses[module][0]
                            loss = loss * args.module_loss_weight / args.gradient_accumulation
                            cont_loss += loss
                    batch_loss += cont_loss
                    class_reps, neg_reps = dict(),  dict()

                batch_loss.backward()
                batch_loss = 0.
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if global_steps % args.report_interval == 0:
                for module in losses:
                    if losses[module]:
                        average_loss = sum(losses[module]) / len(losses[module])
                    else:
                        average_loss = float('nan')
                    tb_writer.add_scalar('loss/%s' % module, average_loss, global_steps)
                tb_writer.add_scalar('lr/lr', scheduler.get_last_lr()[0], global_steps)

            if global_steps % args.evaluate_interval == 0:
                model.eval()
                valid_acc, valid_losses = evaluate_by_module(args, valid_dataloader, model, criterions, preds_file=os.path.join(args.output, args.result_filename) if args.result_filename is not None else None)
                model.train()
                print('writing valid loss to tensorboard...')
                for module, valid_loss in valid_losses.items():
                    tb_writer.add_scalar(
                        'valid/{}'.format(module), valid_loss, global_steps)
                tb_writer.add_scalar('valid/acc', valid_acc, global_steps)
                print('valid acc: %.4lf; valid loss: %.4lf' % (valid_acc, valid_losses['decoder']))
                print('global steps: %d, skipped num: %d' % (global_steps, skipped_num))

                # save the best model
                if valid_acc > best_acc:
                    save_model(os.path.join(args.output, 'best_model'), model, model_config, args)
                    print('saving model: prev valid acc = %.4lf, new valid acc = %.4lf' % (best_acc, valid_acc))
                    best_acc = valid_acc


if __name__ == '__main__':
    args = get_args()
    print(args)
    backup_code(args.output)
    main(args)

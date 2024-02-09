# code of Video Module Networks
import math
import torch
from torch import detach, nn
from transformers import BertModel, BertConfig
from utils.program_parser import nary_mappings as NARY_MAPPINGS
from video_nmn.modules import NAME_TO_MODULE
from video_nmn.dataset import WORDS_TO_KEEP


class VideoNMN(nn.Module):
    def __init__(self, config, debug=False, pretrain_modules=set()):
        super().__init__()
        self.debug = debug
        self.config = config
        self.pretrain_modules = pretrain_modules
        self.submodules = nn.ModuleDict()
        print('pretrain_modules:', pretrain_modules)

        print('creating only l2 norm for contrastive learning')
        self.contrastive_head = L2Normalize()

        type_keywords = {'actions', 'objects', 'relations'}
        name_to_module = NAME_TO_MODULE
        self.words_to_keep = WORDS_TO_KEEP | type_keywords

        for name in name_to_module:
            print('creating %s module...' % name)
            init_params = [config]
            # self.submodules[name] = name_to_module[name](config)
            if name in ['Superlative']:
                init_params.append(self.submodules['Localize'])
            if name in ['Filter', 'Superlative', 'ToAction']:
                init_params.append(self.contrastive_head)
            self.submodules[name] = name_to_module[name](*init_params)

        # 上面写的只使用Linear的encoder不适合建模video和question中的时序关系
        # 因此这里需要一些能进行时序建模的模型
        self.submodules['video_encoder'] = nn.LSTM(
            input_size=config['video_size'], hidden_size=int(config['hidden_size']/2),
            batch_first=True, bidirectional=True
        )

        self.submodules['text_encoder'] = nn.LSTM(
            input_size=config['text_size'], hidden_size=int(config['hidden_size']/2),
            batch_first=True, bidirectional=True
        )

        decoder_size = config['hidden_size'] * 2  # if config['question_encoder'] == 'bert' else int(config['hidden_size'] * 1.5)
        self.submodules['decoder'] = nn.Sequential(
            nn.Linear(decoder_size, decoder_size), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(decoder_size, config['answer_vocab_length'])
        )

        # print number of parameters
        print('number of parameters in the model:')
        total_params = 0
        for name in self.submodules:
            submodule = self.submodules[name]
            num_params = sum([x.numel() for x in submodule.parameters()])
            print('submodule %s has %d parameters' % (name, num_params))
            total_params += num_params
        print('model has %d parameters' % total_params)

    def forward(self, data, return_res_by_step=True, return_result_of_each_step=False, test_mode=False):
        '''
        return_result_of_each_step: used to inspect intermediate result 
        '''
        question, video_feat, prog_str_to_question_tokens, program_list, program_idx = \
            data['question'], data['video_features'], data['prog_str_to_question_tokens'], \
            data['nmn_program_list'], data['nmn_program_idx']

        # encode questions
        video_feat = self.encode_video(video_feat)
        token_feature, question_feature = self.encode_question(question)

        # convert the intermediate results in sg_res_by_step
        new_sg_res_by_step = dict()
        for key, value in data.get('sg_res_by_step', {}).items():
            if isinstance(value, list) and len(value) and isinstance(value[0][1], torch.Tensor):      # value is a list of class namestom
                ans = list()
                for v in value:
                    v_class_name, v_emb = v
                    _, v_emb = self.encode_question_no_grad(v_emb)
                    v_emb = self.contrastive_head(v_emb)
                    ans.append((v_class_name, v_emb))
                new_sg_res_by_step[key] = ans
            else:
                new_sg_res_by_step[key] = value

        if self.debug:
            print('program_list', program_list)

        stack, res_by_step = list(), dict()

        result_of_each_step = list()
        for i in range(len(program_list) - 1, -1, -1):
            prog = program_list[i]
            params = []
            if prog in self.submodules:
                for _ in range(NARY_MAPPINGS[prog]):
                    param = stack.pop()
                    if param == 'video':
                        param = video_feat
                    params.append(param)
                execution_result = self.submodules[prog](*params)
                if return_res_by_step and program_idx[i] is not None and prog in self.pretrain_modules and i != 0:
                    # we should not train the last module, as it should be trained by the decoder
                    if self.config['have_pretrain_head']:
                        pretrain_output = self.submodules[prog].pretrain_head(execution_result)
                    else:
                        pretrain_output = execution_result
                    res_by_step[program_idx[i]] = (prog, pretrain_output)

                if return_result_of_each_step:
                    if self.config['have_pretrain_head'] and prog in self.pretrain_modules:
                        result_of_each_step.append((params, self.submodules[prog].pretrain_head(execution_result)))
                    else:
                        result_of_each_step.append((params, execution_result))

            elif prog in self.words_to_keep:
                execution_result = prog
                if return_result_of_each_step:
                    result_of_each_step.append((params, execution_result))

            else:
                # get the representation of this string from question_feature
                start_token, end_token = prog_str_to_question_tokens[i]
                execution_result = torch.mean(token_feature[start_token: end_token, :], dim=0)
                if return_result_of_each_step:
                    result_of_each_step.append((params, execution_result))

            stack.append(execution_result)        # add dropout here to prevent overfit

        assert len(stack) == 1
        hidden = stack[0]
        hidden_and_ques = torch.cat([hidden, question_feature])
        logits = self.submodules['decoder'](hidden_and_ques)

        ret = {'logits': logits, 'res_by_step': res_by_step}
        if return_result_of_each_step:
            ret['result_of_each_step'] = list(reversed(result_of_each_step))
        if not test_mode:
            ret['sg_res_by_step'] = new_sg_res_by_step
        return ret

    @torch.no_grad()
    def encode_question_no_grad(self, question):
        return self.encode_question(question)

    def encode_question(self, question):
        '''
            question: question word embeddings or token ids
            returns: (token_feature, sent_feature) of shape [question_len, hidden_size], [hidden_size]
        '''
        question = question.unsqueeze(0)
        token_feature, (sent_feature, _) = self.submodules['text_encoder'](question)
        return token_feature[0], sent_feature[:, 0, :].reshape(-1)

    def encode_video(self, video_feat):
        video_feat = video_feat.unsqueeze(0)   # [1 (batch_size), num_frames, video_hidden_size]
        video_feat, _ = self.submodules['video_encoder'](video_feat)
        return video_feat[0]

    def decode(self, hidden, question_feature, first_prog):
        hidden_and_ques = torch.cat([hidden, question_feature])
        if self.individual_decoder:
            if first_prog in ['And', 'Exists', 'Equals', 'Xor']:
                logits = self.submodules['decoder']['yes_no'](hidden_and_ques)
            elif first_prog == 'Compare':
                logits = self.submodules['decoder']['before_after'](hidden_and_ques)
            else:
                logits = self.submodules['decoder']['open_ended'](hidden_and_ques)
        else:
            logits = self.submodules['decoder'](hidden_and_ques)
        return logits


def format_params(param):
    if isinstance(param, torch.Tensor):
        return param.size() if param.numel() > 50 else param
    elif isinstance(param, (list, tuple)):
        return [format_params(i) for i in param]
    elif isinstance(param, dict):
        return {key: format_params(value) for key, value in param.items()}
    else:
        return param


def create_attention_from_frame_interval(frame_interval, pred):
    gold_tensor = torch.zeros_like(pred)
    num_frames = pred.size(-1)
    # if isinstance(frame_interval[0], float):        # Temporal module
    #     start, end = max(0.001, frame_interval[0]), min(num_frames - 0.001, frame_interval[1])
    #     start_int, end_int = math.ceil(start), math.floor(end)
    #     if start_int < end_int:
    #         gold_tensor[start_int:end_int] += 1
    #     gold_tensor[start_int - 1] += start_int - start
    #     gold_tensor[end_int] += end - end_int
    # if isinstance(frame_interval[0], tuple) and isinstance(frame_interval[0][0], float):      # Localize module
    for i in range(pred.size(0)):
        start, end = max(0.001, frame_interval[i][0]), min(num_frames - 0.001, frame_interval[i][1])
        start_int, end_int = math.ceil(start), math.floor(end)      # e.g., (0.2, 5.8) -> (1, 5)
        if start_int < end_int:
            gold_tensor[i, start_int:end_int] += 1      # gold[1:5] = 1
        gold_tensor[i, start_int-1] += start_int - start    # gold[0] = 1 - 0.2
        gold_tensor[i, end_int] += end - end_int            # gold[5] = 5.8 - 5
    return gold_tensor


class L2Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        return nn.functional.normalize(feat, dim=0)

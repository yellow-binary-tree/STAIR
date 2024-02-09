import torch
from torch import nn

from utils.program_parser import nary_mappings as NARY_MAPINGS


class AndModule(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, feat1, feat2):
        return torch.min(feat1, feat2)


class CompareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(nn.Linear(config['hidden_size']*2, config['hidden_size']), nn.ReLU())

    def forward(self, feat1, feat2):
        return self.param(torch.cat([feat1, feat2]))


class EqualsModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(nn.Linear(config['hidden_size']*2, config['hidden_size']), nn.ReLU())
        if config['have_pretrain_head']:
            self.pretrain_head = nn.Linear(config['hidden_size'], 1)

    def forward(self, feat1, feat2):
        '''
        feat1: [hidden_size]
        feat2: [hidden_size]
        returns: scalar tensor, the two feats are same or not
        '''
        return self.param(torch.cat([feat1, feat2]))


class ChooseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, keyword1, keyword2, query):
        '''
        keyword1: [hidden_size]
        keyword2: [hidden_size]
        query: [hidden_size]
        returns: keyword1 or keyword2, decided by cos sim with query. [hidden_size]
        '''
        if self.cosine_similarity(keyword1.unsqueeze(0), query.unsqueeze(0)) > \
            self.cosine_similarity(keyword2.unsqueeze(0), query.unsqueeze(0)):
            return keyword1
        else:
            return keyword2


class XorModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(nn.Linear(config['hidden_size']*3, config['hidden_size']), nn.ReLU())
        if config['have_pretrain_head']:
            self.pretrain_head = nn.Linear(config['hidden_size'], 2)

    def forward(self, feat1, feat2):
        '''
        feat1: [hidden_size]
        feat2: [hidden_size]
        returns: [hidden_size]
        '''
        return self.param(torch.cat([torch.abs(feat1-feat2), feat1, feat2]))


class XorFrameModule(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, feat1, feat2):
        return torch.abs(feat1 - feat2)


class QueryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
        )

        if config['have_pretrain_head']:
            self.pretrain_head = nn.Linear(config['hidden_size'], config['object_types'])

    def forward(self, keyword):
        '''
        keyword: [hidden_size]
        returns: [hidden_size]
        '''
        res = self.param(keyword)
        return res


class ToActionModule(nn.Module):
    def __init__(self, config, contrastive_head=None):
        super().__init__()
        self.param = nn.Sequential(
            nn.Linear(config['hidden_size']*2, config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU()
        )

        if config['have_pretrain_head']:
            self.pretrain_head = contrastive_head

    def forward(self, action, keyword):
        '''
        action: [hidden_size]
        keyword: [hidden_size]
        returns: [hidden_size]
        '''
        res = self.param(torch.cat([action, keyword]))
        return res


class HasItemModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], 1), nn.Sigmoid(), nn.Dropout(config['dropout']),
        )
        if config['have_pretrain_head']:
            self.pretrain_head = nn.Identity()

    def forward(self, feat):
        '''
        feat: [num_frames, hidden_size]
        returns: [num_frames], the plausibility of having an item in each frame
        '''
        return self.param(feat).squeeze()


class ExistsModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.Sequential(
            nn.Linear(config['hidden_size']*3, config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
        )

        if config['have_pretrain_head']:
            self.pretrain_head = nn.Linear(config['hidden_size'], 2)

    def forward(self, keyword, feat):
        '''
        keyword: [hidden_size]
        feat: [hidden_size]
        returns: [hidden_size], whether keyword exists in feat
        '''
        res = self.param(torch.cat([feat, keyword, feat*keyword]))
        return res


class ExistsFrameModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.CosineSimilarity(dim=-1)
        if config['have_pretrain_head']:
            self.pretrain_head = nn.Identity()

    def forward(self, keyword, feat):
        '''
        keyword: [hidden_size]
        feat: [num_frames, hidden_size]
        returns: a attention, whether keyword exists in feat [num_frames]
        '''
        keyword = keyword.unsqueeze(0)
        attention_scores = self.attention(feat, keyword)      # [num_kws, num_frames]
        attention_scores = (attention_scores + 1) * 0.49      # * 0.49 is to avoid nan in CE loss
        return attention_scores


class LocalizeModule(nn.Module):
    '''
    predicts the relevant frames of a given action
    '''
    def __init__(self, config):
        super().__init__()
        self.video_linear = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], config['hidden_size']),
        )
        self.keyword_linear = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
        )
        self.attention = nn.CosineSimilarity(dim=-1)

        if config['have_pretrain_head']:
            self.pretrain_head = nn.Identity()

    def forward(self, feat, keyword):
        '''
        feat: [num_frames, hidden_size]
        keyword: [num_kws, hidden_size]
        returns: attention map of all actions, [num_kws, num_frames]
        '''
        feat = self.video_linear(feat)
        if len(keyword.size()) == 1:
            keyword = keyword.unsqueeze(0)
        keyword = self.keyword_linear(keyword)

        feat = feat.unsqueeze(0)
        keyword = keyword.unsqueeze(1)
        feat_expand = feat.expand(keyword.size(0), feat.size(1), feat.size(2))
        keyword_expand = keyword.expand(keyword.size(0), feat.size(1), feat.size(2))        # [num_kws, num_frames, hidden_size]

        attention_scores = self.attention(feat_expand, keyword_expand)      # [num_kws, num_frames]
        attention_scores = (attention_scores + 1) * 0.49                # * 0.49 is to avoid nan in CE loss
        return attention_scores


class SuperlativeModule(nn.Module):
    def __init__(self, config, localize_module=None, contrastive_head=None):
        super().__init__()
        if localize_module is None:
            self.localize_module = LocalizeModule(config)
        else:
            self.localize_module = localize_module
        self.softmax = nn.Softmax(dim=0)
        self.dense = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), # nn.Dropout(config['dropout']),
        )

        if config['have_pretrain_head']:
            self.pretrain_head = contrastive_head

    def forward(self, mode, actions, feat):
        '''
        mode: one of {max, min}
        actions: [num_actions, hidden_size]
        attention_scores: [num_actions, num_frames]
        feat: [num_frames, hidden_size]
        returns: a action, [hidden_size]
        '''
        attention_scores = self.localize_module(feat, actions)
        weight = self.softmax(attention_scores.sum(dim=1))      # [num_actions]
        if mode == 'min':
            weight = 1 - weight     # 因为理论上actions应该只有两个，但实际上也不重要了，反正这个模块的覆盖面很小
        output_action = self.dense(torch.sum(weight.unsqueeze(1) * actions, dim=0))
        return output_action


class TemporalModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['max_video_length'] > 32:
            print('the video is long, so use Conv1d for temporal reasoning')
            self.nn_mode = 'conv'
            self.kernel_size = round(config['max_video_length'] / 4)
            padding_mode = 'zeros'
            self.relate = nn.ModuleDict({
                mode: nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding='same', padding_mode=padding_mode), nn.ReLU(),
                    nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding='same', padding_mode=padding_mode), nn.ReLU(),
                    nn.Conv1d(1, 1, kernel_size=self.kernel_size*2+1, padding='same', padding_mode=padding_mode), nn.Sigmoid()
                ) for mode in ['before', 'after', 'between']
            })
        else:
            print('the video is short and at same length, so use Linear for temporal reasoning')
            self.max_video_length = config['max_video_length']
            self.nn_mode = 'linear'
            self.relate = nn.ModuleDict({
                mode: nn.Sequential(
                    nn.Linear(config['max_video_length'], config['max_video_length']), nn.ReLU(),
                    nn.Linear(config['max_video_length'], config['max_video_length']), nn.ReLU(),
                    nn.Linear(config['max_video_length'], config['max_video_length']), nn.Sigmoid(),
                ) for mode in ['before', 'after', 'between']
            })
        self.relate['while'] = nn.Identity()

        self.dense = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
        )
        self.layer_norm = nn.LayerNorm(config['hidden_size'])
        self.relu = nn.ReLU()
        self.related_attn = None

    def pretrain_head(self, *args):
        return self.related_attn

    def relate_(self, attention_scores, mode):
        '''
        attention_scores: [num_frames], or [2, num_frames] if mode == `between`
        returns: [num_frames]
        '''
        if mode == 'while':
            return attention_scores.squeeze()
        attention_scores = self.relu(attention_scores).squeeze()
        if mode == 'before':
            ret_attn = torch.cumsum(attention_scores, dim=-1)
        if mode == 'after':
            ret_attn = torch.cumsum(attention_scores.flip([-1]), dim=-1).flip([-1])
        if mode == 'between':
            att_before_a = self.relate_(attention_scores[0], 'before')
            att_after_a = self.relate_(attention_scores[0], 'after')
            att_before_b = self.relate_(attention_scores[1], 'before')
            att_after_b = self.relate_(attention_scores[1], 'after')
            ret_attn = torch.max(torch.min(att_before_a, att_after_a), torch.min(att_before_b, att_after_b))
        return ret_attn

    def forward(self, mode, feat, attention_scores):
        '''
        mode: str, one of 'while', 'before', 'after', 'between'
        feat: [num_frames, hidden_size]
        # actions: if mode == 'between' then [2, hidden_size] else [hidden_size]
        attention_scores: [1 or 2, num_frames]
        returns: new feat, [num_frames, hidden_size]
        '''

        attention_scores = torch.mean(attention_scores, dim=0)
        if self.nn_mode == 'conv':
            attention_scores = attention_scores.view(1, 1, -1)
            ret_attn = self.relate[mode](attention_scores)          # [1(bs), 1(out_c), num_frames]
            self.related_attn = ret_attn.squeeze()
        else:
            self.related_attn = self.relate[mode](attention_scores)
        output_feat = self.dense(self.related_attn.unsqueeze(-1) * feat)
        return self.layer_norm(output_feat)


class AttnVideoModule(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, feat, attn):
        '''
        feat: [num_frames, hidden_size]
        attn: [num_frames]
        returns: attn * feat [num_frames, hidden_size]
        '''
        return attn.unsqueeze(1) * feat


class FilterModule(nn.Module):
    def __init__(self, config, contrastive_head=None):
        super().__init__()
        self.param = nn.ModuleDict({
            kw: nn.Sequential(
                nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
                nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            ) for kw in ['representation', 'actions', 'objects', 'relations']
        })

        self.attention = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, 1), nn.Softmax(),
        )
        self.dense = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), # nn.Dropout(config['dropout']),
        )

        if config['have_pretrain_head']:
            self.pretrain_head = contrastive_head

    def forward(self, feat, keyword):
        '''
        feat: [num_frames, hidden_size]
        keyword: a text item, [hidden_size]
        returns: a represenation in text space, [hidden_size]
        '''
        if isinstance(keyword, torch.Tensor):
            feat = self.param['representation'](feat)
            keyword = keyword.unsqueeze(0)
            feat_and_keyword = torch.cat([feat, keyword.expand(feat.size(0), keyword.size(1))], dim=1)    # [num_frames, hidden_size*2]
            attention_scores = self.attention(feat_and_keyword)
            aggregated_feature = torch.sum(attention_scores * feat, dim=0)      # [hidden_size]
        else:
            aggregated_feature = torch.sum(self.param[keyword](feat), dim=0)
        ret_feat = self.dense(aggregated_feature)
        return ret_feat


class FilterFrameModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.param = nn.ModuleDict({kw: nn.Sequential(
                nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
                nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
            ) for kw in ['representation', 'relations', 'actions']
        })
        self.attention = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, 1), nn.Sigmoid(),
        )
        self.dense = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']), nn.ReLU(), nn.Dropout(config['dropout']),
        )

        if config['have_pretrain_head']:
            self.pretrain_head = nn.Linear(config['hidden_size'], config['object_types'])

    def forward(self, feat, keyword):
        '''
        feat: [num_frames, hidden_size]
        keyword: a text item, [hidden_size]
        returns: a hidden state for each frame, [num_frames, hidden_size]
        '''
        if isinstance(keyword, torch.Tensor):
            feat = self.param['representation'](feat)
            keyword = keyword.unsqueeze(0)
            feat_and_keyword = torch.cat([feat, keyword.expand(feat.size(0), keyword.size(1))], dim=1)    # [num_frames, hidden_size*2]
            attention_scores = self.attention(feat_and_keyword)
            aggregated_feature = attention_scores * feat
        else:
            aggregated_feature = self.param[keyword](feat)
        ret_feat = self.dense(aggregated_feature)
        return ret_feat


class RelateModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = nn.Parameter(torch.rand(config['max_video_length']))
        self.softmax = nn.Softmax()

    def forward(self, mode, attn):
        '''
        mode: a text choice in {forward, backward}
        attn: [num_frames]
        returns: a shifted attn, [num_frames]
        '''
        mode = 'fwd' if mode == 'forward' else 'bak'
        num_frames = attn.size(0)
        if mode == 'fwd':
            attn = attn + self.beta[:num_frames]
        else:
            attn = attn - self.beta[:num_frames]
        return self.softmax(attn)


class Array2Module(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def __call__(self, feat1, feat2):
        return torch.stack([feat1, feat2])


NAME_TO_MODULE = {
    'And': AndModule,
    'AttnVideo': AttnVideoModule,
    'Choose': ChooseModule,
    'Compare': CompareModule,
    'Equals': EqualsModule,
    'Exists': ExistsModule,
    'ExistsFrame': ExistsFrameModule,
    'Filter': FilterModule,
    'FilterFrame': FilterFrameModule,
    'HasItem': HasItemModule,
    'Localize': LocalizeModule,
    'Relate': RelateModule,
    'Superlative': SuperlativeModule,
    'Temporal': TemporalModule,
    'ToAction': ToActionModule,
    'Xor': XorModule,
    'XorFrame': XorFrameModule,
    'Array2': Array2Module,
}

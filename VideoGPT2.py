
from transformers import GPT2Model, GPT2PreTrainedModel, AutoModelForCausalLM, PreTrainedModel, LlamaForCausalLM, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D, prune_conv1d_layer
import logging
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        #w = w * b - 1e18 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            b = torch.gt(b + attention_mask[0], 0).float()      # 让一些token可以一直被看见
            w = w * b - 1e18 * (1 - b)
            w = w - 1e18 * (1 - attention_mask[1])      # mask掉paddings
        else:
            w = w * b - 1e18 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class VideoGPT2Model(GPT2Model):
    def __init__(self, config):
        super(VideoGPT2Model, self).__init__(config)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        config_dict = config.to_dict()
        self.output_hidden_states = config_dict.get('output_hidden_states', False)
        self.output_attentions = config_dict.get('output_attentions', False)

    def forward(self, input_embs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_embs.size(-2) + past_length, dtype=torch.long, device=input_embs.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_embs[:, :, 0])

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask[0] = attention_mask[0].unsqueeze(1).unsqueeze(2)
            attention_mask[1] = attention_mask[1].unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask[0] = attention_mask[0].to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask[1] = attention_mask[1].to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            #attention_mask = (1.0 - attention_mask) * -1e18

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_embs.size()[:2]
        # input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        # inputs_embeds = self.wte(input_ids)
        inputs_embeds = input_embs
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


# class VideoGPT2LMHeadModel(GPT2PreTrainedModel):
class LMHeadModel(PreTrainedModel):
    def __init__(self, config, model_path, lm_model='VideoGPT'):
        super(LMHeadModel, self).__init__(config)
        self.base_model_prefix = 'transformer'
        self.lm_model = lm_model
        if not hasattr(config, 'n_embd') and hasattr(config, 'hidden_size'):
            config.n_embd = config.hidden_size
        config.tie_word_embeddings = True
        self.config = config
        if lm_model == 'VideoGPT':
            self.transformer = VideoGPT2Model.from_pretrained(model_path, config=config)
            # self.transformer = GPT2LMHeadModel.from_pretrained(model_path, config=config)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.get_output_embeddings = self.get_output_embeddings
        else:
            self.transformer = LlamaForCausalLM.from_pretrained(model_path, config=config)

        # if not hasattr(self.transformer, 'wte'):    # and hasattr(self.transformer, 'embed_tokens'):
            # self.transformer.wte = self.transformer.get_input_embeddings()

        self.video_ff = nn.Linear(config.video_feature_dim, config.n_embd)
        self.video_inverse_ff = nn.Linear(config.n_embd, config.video_feature_dim)

        if self.config.llm_lora:
            self.lora_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            self.transformer = get_peft_model(self.transformer, self.lora_config)
            logging.info('number of trainable LORA parameters in LLM:')
            self.transformer.print_trainable_parameters()

        self.init_weights()     # will not init weights loaded by from_pretrained() again
        self.tie_weights()        # 如果我们直接加载for causal lm模型，就不用自己设定lm head了

        logging.info('model_config: {}'.format(config))
        # logging.info('model_structure: {}'.format(self))

    def get_output_embeddings(self):
        if hasattr(self, 'lm_head'):
            return self.lm_head
        else:
            return self.transformer.get_output_embeddings()

    def get_llm_lora_params(self):
        if not self.config.llm_lora:
            return dict()
        return get_peft_model_state_dict(self.transformer)

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.transformer.get_output_embeddings(), self.transformer.get_input_embeddings())

        # if hasattr(self.transformer, 'wte'):
        #     self._tie_or_clone_weights(self.lm_head, self.transformer.wte)
        # elif hasattr(self.transformer, 'embed_tokens'):
        #     self._tie_or_clone_weights(self.lm_head, self.transformer.embed_tokens)
        # else:
        #     raise ValueError("Can not find a word embedding parameter. Please specify it")

    def state_dict(self):
        # print('getting state dict')
        # all_state_dict = super().state_dict()
        ret = {name: param for name, param in self.named_parameters() if param.requires_grad}
        return ret

    # @torch.cuda.amp.autocast()
    # def forward(self, input_embs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, mode="reply"):
    @torch.cuda.amp.autocast(dtype=torch.float16)
    def forward(self, input_ids=None, video_feats=None, input_embs=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, video_lengths=None, mode="reply"):
        if input_embs is None:
            input_embs = self.transformer.get_input_embeddings()(input_ids)
            if video_feats is not None:
                # print('{}: {} - {}'.format('video_feats', video_feats.size(), video_feats.dtype))
                # print('{}: {}'.format('self.video_ff', self.video_ff.weight.dtype))
                video_embs = self.video_ff(video_feats)
                input_embs = torch.cat([video_embs, input_embs], dim=1)

        if self.lm_model == 'VideoGPT':
            transformer_outputs = self.transformer(input_embs,
                                                # past=past,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids)
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

        elif self.lm_model == 'Llama':
            transformer_outputs = self.transformer(inputs_embeds=input_embs,
                                                   position_ids=position_ids,
                                                   return_dict=True)
            hidden_states = transformer_outputs.hidden_states
            lm_logits = transformer_outputs.logits

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            if mode == "reply":
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[0][..., 1:].contiguous()
                # Flatten the tokens
                loss_text_fct = CrossEntropyLoss(ignore_index=-1)
                loss_text = loss_text_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
                loss = loss_text
            else:       # mode == 'video'
                video_feat = labels[1]
                lm_video_regs = self.video_inverse_ff(hidden_states[:, :max(video_lengths), :])

                loss = 0.
                loss_video_fct = MSELoss(reduce=True, size_average=True)
                for i, (v_label, v_pred, v_len) in enumerate(zip(video_feat, lm_video_regs, video_lengths)):
                    v_label = v_label[1:]
                    loss_video = loss_video_fct(v_pred[:v_len - 1], v_label[:v_len - 1])
                    loss += loss_video
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

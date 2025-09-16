from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import transformers
from datetime import datetime
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, \
    _prepare_4d_causal_attention_mask_for_sdpa
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM

from transformers.utils import ModelOutput
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

import subprocess

import torch
import random
import numpy as np
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True


class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert: nn.Module,
        num_experts: int,
        ep_size: int = 1,  
        k: int = 1,        
        capacity_factor: float = 1.0,      
        eval_capacity_factor: float = 1.0, 
        min_capacity: int = 4,             
        use_residual: bool = False,       
    ):
        super(MoE, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.use_residual = use_residual

        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])

        if self.use_residual:
            self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_size = x.size()
        x_flat = x.view(-1, hidden_size)  
        num_tokens = x_flat.size(0)

        gate_scores = self.gate(x_flat)  
        if gate_scores.shape != (1, 4):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = f"pt_files/gate_scores_{timestamp}.pt"
            torch.save(gate_scores, file_path)

        top_k_scores, top_k_indices = gate_scores.topk(self.k, dim=-1)  # [num_tokens, k]
        top_k_weights = F.softmax(top_k_scores, dim=-1)  # [num_tokens, k]

        capacity_factor = self.capacity_factor if self.training else self.eval_capacity_factor
        capacity = max(self.min_capacity, int((num_tokens * self.k) / self.num_experts * capacity_factor))

        output = torch.zeros_like(x_flat)
        exp_counts = torch.zeros(self.num_experts, dtype=torch.int64, device=x.device)

        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)  
            expert_weights = top_k_weights * expert_mask.float()  
            expert_weight_per_token = expert_weights.sum(dim=-1)  
            mask = expert_weight_per_token > 0  
            exp_counts[i] = mask.sum().item()

            if mask.sum() > 0:
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                weights = expert_weight_per_token[mask].unsqueeze(-1)  
                output[mask] += expert_output * weights

        if self.use_residual:
            output = output + self.residual_weight * x_flat
        output = output.view(batch_size, seq_len, hidden_size)

        expert_load = gate_scores.mean(dim=0)
        aux_loss = (expert_load ** 2).sum()

        return output, aux_loss, exp_counts


class MoELlavaQwenConfig(Qwen2Config):
    model_type = "moe_llava_qwen"

    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[
                # 'up_proj', 'down_proj', 'gate_proj', 'wg',
                # 'embed_tokens', 'lm_head'
            ]
        )
        self.lora = {}

        super(MoELlavaQwenConfig, self).__init__(**kwargs)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


# MoE-enabled model class
class MoELlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = MoELlavaQwenConfig

    def __init__(self, config: MoELlavaQwenConfig):
        super(MoELlavaQwenModel, self).__init__(config)

# Modified forward function for Qwen2DecoderLayer to support MoE
def MoEQwen2DecoderLayer_forward(layer):
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]:
        
        # # import pdb; pdb.set_trace()
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (MLP with potential MoE)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        # 1706 M
        mlp_output = layer.mlp(hidden_states)
        
        moe_losses = []
        if isinstance(mlp_output, tuple):  # MoE MLP returns (output, loss)
            hidden_states = mlp_output[0]
            moe_losses.append(mlp_output[1])
        else:
            hidden_states = mlp_output

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs
    return forward

# Modified forward function for Qwen2Model to collect MoE losses
def MoEQwen2Model_forward(model):
    def forward(
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_moe_loss: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else model.config.use_cache
        return_dict = return_dict if return_dict is not None else model.config.use_return_dict

        # import pdb; pdb.set_trace()
        # ... (input processing logic from Qwen2Model.forward)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        if inputs_embeds is None:
            inputs_embeds = model.embed_tokens(input_ids)


        if attention_mask is not None and model._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if model._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif model._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=model.config.sliding_window,
            )
            
        all_moe_losses = [] if output_moe_loss else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        hidden_states = inputs_embeds
        for decoder_layer in model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if all_moe_losses is not None and len(layer_outputs) > (2 if use_cache else 1) + (1 if output_attentions else 0):
                all_moe_losses.append(layer_outputs[-1])
        
        # import pdb; pdb.set_trace()
        hidden_states = model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns, all_moe_losses] if v is not None)

        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_losses,
        )
    return forward

# MoE-enabled CausalLM class
class MoELlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = MoELlavaQwenConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "moe_llava_qwen"
        config.rope_scaling = None

        self.model = MoELlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def initialize_moe_modules(self, model_args):
        """Initialize MoE layers based on config settings."""

        self.config.lora = {}
        self.config.moe = {}
        
        if getattr(model_args, 'lora_enable', False):
            
            self.config.lora['lora_enable'] = model_args.lora_enable
            self.config.lora['only_lora_ffn'] = model_args.only_lora_ffn
            self.config.lora['lora_r'] = model_args.lora_r
            self.config.lora['lora_alpha'] = model_args.lora_alpha
            self.config.lora['lora_dropout'] = model_args.lora_dropout
            self.config.lora['lora_bias'] = model_args.lora_bias
            # self.config.lora['modules_to_save'] = model_args.modules_to_save
            self.config.lora['target_modules'] = model_args.train_modules

        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size']= model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        # self.config.moe['train_modules'] = [
        #         # 'mlp.w1', 'mlp.w2', 'mlp.c_proj', 'wg',
        #         # 'wte', 'lm_head'
        #     ]
        
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe['train_modules']):
                    continue
                else:
                    p.requires_grad = False
        

        moe_config = self.config.moe
        num_layers = self.config.num_hidden_layers
        
        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {model_args.moe_mode}')


        self.config.moe['moe_layers_idx'] = moe_layers_idx
        # if len(model_args.num_experts) == 1:
        self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        # assert len(self.config.moe['num_experts']) == len(moe_layers_idx)



        moe_layers_idx = moe_config.get('moe_layers_idx', [])

        if not moe_layers_idx:
            moe_mode = moe_config.get('moe_mode', 'dense')
            if moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif moe_mode == "sparse":
                moe_layers_idx = list(range(0, num_layers, 2))
            elif moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(f"Unsupported moe_mode: {moe_mode}")
            moe_config['moe_layers_idx'] = moe_layers_idx

        num_experts_list = moe_config.get('num_experts', [2] * len(moe_layers_idx))  # Default to 2 experts per layer

        if len(num_experts_list) == 1:
            num_experts_list = num_experts_list * len(moe_layers_idx)
        assert len(num_experts_list) == len(moe_layers_idx), "Number of experts must match number of MoE layers"

        # Replace MLP with MoE in specified layers
        
        for layer_idx, num_experts in zip(moe_layers_idx, num_experts_list):
            layer = self.model.layers[layer_idx]

            layer.mlp = MoE(
                hidden_size=self.config.hidden_size,
                expert=layer.mlp,
                num_experts=num_experts,
                ep_size=moe_config.get('ep_size', 1),
                k=moe_config.get('top_k_experts', 1),
                capacity_factor=moe_config.get('capacity_factor', 1.0),
                eval_capacity_factor=moe_config.get('eval_capacity_factor', 1.0),
                min_capacity=moe_config.get('min_capacity', 4),
                use_residual=moe_config.get('use_residual', False),
            )

        # Modify forward methods to handle MoE
        for layer in self.model.layers:
            layer.forward = MoEQwen2DecoderLayer_forward(layer)
        self.model.forward = MoEQwen2Model_forward(self.model)

        print(f"Initialized MoE: {len(moe_layers_idx)} layers with experts: {num_experts_list}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes
            )
        # import pdb; pdb.set_trace()
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        # Run the transformer model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)


        loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_loss, moe_losses = None, []
        if return_dict:
            moe_loss_list = outputs.moe_loss_list if outputs.moe_loss_list is not None else []
        else:
            moe_loss_list = outputs[-1] if len(outputs) > 4 else [] 

        if len(moe_loss_list) > 0:
            for moe_loss in moe_loss_list:
                if moe_loss is not None and moe_loss != []:
                    moe_losses.append(moe_loss)
            moe_losses = [moe_loss[0] for moe_loss in moe_losses if moe_loss != []]
            moe_loss = self.router_aux_loss_coef * sum(moe_losses)
            if labels is not None:
                print(loss, sum(moe_losses), loss + moe_loss)
                loss += moe_loss
                
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            if moe_loss is not None:
                output = (moe_loss,) + output
            return ((loss,) + output) if loss is not None else output
        # import pdb; pdb.set_trace()
        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=outputs.moe_loss_list,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        # # import pdb; pdb.set_trace()
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        
        # # import pdb; pdb.set_trace()
        
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        
        inputs["past_key_values"] = past_key_values
        
        return inputs

class EvalMoELlavaQwenForCausalLM(MoELlavaQwenForCausalLM):
    config_class = MoELlavaQwenConfig
    
    def __init__(self, config):
        super(EvalMoELlavaQwenForCausalLM, self).__init__(config)
        
        if getattr(self.config, 'lora', False) and self.config.lora.get('lora_enable', False):
            from peft import LoraConfig, get_peft_model
            pre_lora_config = self.config.lora
            lora_config = LoraConfig(
                r=pre_lora_config['lora_r'],
                lora_alpha=pre_lora_config['lora_alpha'],
                target_modules=pre_lora_config['target_modules'],
                lora_dropout=pre_lora_config['lora_dropout'],
                bias=pre_lora_config['lora_bias'],
                # modules_to_save=pre_lora_config['modules_to_save'],
                task_type="CAUSAL_LM",
            )
            # print("Adding LoRA adapters...")
            # get_peft_model(self, lora_config)

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']

        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            self.model.layers[layer_num].mlp = MoE(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=self.config.moe['ep_size'],
                k=self.config.moe['top_k_experts'],
                capacity_factor=self.config.moe['capacity_factor'],
                eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                min_capacity=self.config.moe['min_capacity'],
                use_residual=self.config.moe['use_residual'],
            )
        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEQwen2DecoderLayer_forward(m)
        print(f'replace Qwen1_5DecoderLayer.forward to MoEQwen1_5DecoderLayer.forward')
        self.model.forward = MoEQwen2Model_forward(self.model)
        print(f'replace Qwen1_5Model.forward to MoEQwen1_5Model.forward')


AutoConfig.register("moe_llava_qwen", MoELlavaQwenConfig)
AutoModelForCausalLM.register(MoELlavaQwenConfig, MoELlavaQwenForCausalLM)
AutoModelForCausalLM.register(MoELlavaQwenConfig, EvalMoELlavaQwenForCausalLM)

import os
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union
from abc import ABC, abstractmethod
from utils.misc import ignore_warnings
from transformers import AutoConfig, AutoModelForCausalLM
from utils.constants import IGNORE_INDEX, VIDEO_TOKEN_INDEX
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM


class PiTeMetaModel:
    def initialize_mlp_modules(self, model_args):
        def get_w(weights, keyword):
            return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

        if not hasattr(self, "vision_adapter"):
            self.vision_adapter = nn.Linear(768, self.config.hidden_size)
        if model_args.pretrain_vision_adapter is not None:
            vision_adapter_weights = torch.load(model_args.pretrain_vision_adapter, map_location="cpu")
            self.vision_adapter.load_state_dict(get_w(vision_adapter_weights, "vision_adapter"))
            print(f"\033[33mLoad Vision Adapter: {model_args.pretrain_vision_adapter} ...\033[0m")
        
        if not hasattr(self, "localization_projector"):
            self.localization_projector = nn.Linear(self.config.hidden_size, 2)
        if hasattr(model_args, "pretrain_localization_projector"):
            localization_projector_weights = torch.load(model_args.pretrain_localization_projector, map_location="cpu")
            self.localization_projector.load_state_dict(get_w(localization_projector_weights, "localization_projector"))
            print(f"\033[33mLoad Localization Projector: {model_args.pretrain_localization_projector} ...\033[0m")
        
        if not hasattr(self, "trajectory_projector"):
            self.trajectory_projector = nn.Linear(self.config.hidden_size, 3 * 100 * 2)
        if hasattr(model_args, "pretrain_localization_projector"):
            localization_projector_weights = torch.load(model_args.pretrain_localization_projector, map_location="cpu")
            localization_projector_state_dict = get_w(localization_projector_weights, "localization_projector")
            trajectory_projector_state_dict = {
                "weight": localization_projector_state_dict["weight"].repeat(3 * 100, 1).contiguous()
                , "bias": localization_projector_state_dict["bias"].repeat(3 * 100).contiguous()
            }
            self.trajectory_projector.load_state_dict(trajectory_projector_state_dict)
            print(f"\033[33mLoad Trajectory Projector From Localization Projector: {model_args.pretrain_localization_projector} ...\033[0m")


class PiTeMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass
    
    def prepare_inputs_labels_for_video(
        self
        , input_ids: torch.LongTensor = None
        , attention_mask: Optional[torch.Tensor] = None
        , position_ids: Optional[torch.LongTensor] = None
        , past_key_values: Optional[torch.FloatTensor] = None
        , labels: Optional[torch.LongTensor] = None
        , videos: Optional[torch.FloatTensor] = None
        , localization: Optional[torch.FloatTensor] = None
        , trajectory_adjs: Optional[torch.FloatTensor] = None
    ) -> tuple:
        if videos is None or input_ids.size(1) == 1:
            if past_key_values is not None and videos is not None and input_ids.size(1) == 1:
                target_shape = past_key_values[-1][-1].size(-2) + 1
            attention_mask = torch.cat(
                tensors=(
                    attention_mask
                    , torch.ones(
                        size=(attention_mask.size(0), target_shape - attention_mask.size(1))
                        , dtype=attention_mask.dtype
                        , device=attention_mask.device
                    )
                )
                , dim=1
            )
            return input_ids, attention_mask, position_ids, past_key_values, None, labels
        if isinstance(videos, list):
            concat_videos = torch.cat(videos, dim=0)
            video_feature = self.get_model().vision_adapter(concat_videos)
            split_sizes = [video.size(0) for video in videos]
            video_feature = torch.split(video_feature, split_sizes, dim=0)
        else:
            video_feature = self.get_model().vision_adapter(videos)
        _labels, _localization, _trajectory_adjs, _position_ids, _attention_mask = labels, localization, trajectory_adjs, position_ids, attention_mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device) if position_ids is None else position_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels
        trajectory_adjs = torch.ones(size=input_ids.size()[:2] + (3, 100, 2), device=input_ids.device) if trajectory_adjs is None else trajectory_adjs
        localization = torch.ones(size=input_ids.size()[:2] + (2, ), device=input_ids.device) if localization is None else localization
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        trajectory_adjs = [cur_trajectory_adjs[cur_attention_mask] for cur_trajectory_adjs, cur_attention_mask in zip(trajectory_adjs, attention_mask)]
        localization = [cur_localization[cur_attention_mask] for cur_localization, cur_attention_mask in zip(localization, attention_mask)]
        new_input_embeds, new_labels, new_localization, new_trajectory_adjs, cur_video_idx = [], [], [], [], 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            n_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
            if n_videos == 0:
                cur_video_features = video_feature[cur_video_idx]
                cur_input_embeds = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds, cur_video_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_localization.append(localization[batch_idx])
                new_trajectory_adjs.append(trajectory_adjs[batch_idx])
                cur_video_idx += 1
                continue
            video_token_indices = [-1] + torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.size(0)]
            cur_input_ids_noim, cur_labels, cur_labels_noim, cur_localization, cur_localization_noim, cur_trajectory_adjs, cur_trajectory_adjs_noim = \
                [], labels[batch_idx], [], localization[batch_idx], [], trajectory_adjs[batch_idx], []
            for i in range(len(video_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[video_token_indices[i] + 1:video_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[video_token_indices[i] + 1:video_token_indices[i + 1]])
                cur_localization_noim.append(cur_localization[video_token_indices[i] + 1:video_token_indices[i + 1]])
                cur_trajectory_adjs_noim.append(cur_trajectory_adjs[video_token_indices[i] + 1:video_token_indices[i + 1]])
            split_sizes = [x.size(0) for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_noim = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds, cur_new_labels, cur_new_localization, cur_new_trajectory_adjs = [], [], [], []
            for i in range(n_videos + 1):
                cur_new_input_embeds.append(cur_input_embeds_noim[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_localization.append(cur_localization_noim[i])
                cur_new_trajectory_adjs.append(cur_trajectory_adjs_noim[i])
                if i < n_videos:
                    cur_video_features = video_feature[cur_video_idx]
                    cur_video_idx += 1
                    cur_new_input_embeds.append(cur_video_features)
                    cur_new_labels.append(torch.full(
                        size=(cur_video_features.size(0), )
                        , fill_value=IGNORE_INDEX
                        , device=cur_labels.device
                        , dtype=cur_labels.dtype
                    ))
                    cur_new_localization.append(torch.full(
                        size=(cur_video_features.size(0), 2)
                        , fill_value=-1
                        , device=cur_labels.device
                        , dtype=cur_labels.dtype
                    ))
                    cur_new_trajectory_adjs.append(torch.full(
                        size=(cur_video_features.size(0), 3, 100, 2)
                        , fill_value=-1
                        , device=cur_labels.device
                        , dtype=cur_labels.dtype
                    ))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_localization = torch.cat(cur_new_localization)
            cur_new_trajectory_adjs = torch.cat(cur_new_trajectory_adjs)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_localization.append(cur_new_localization)
            new_trajectory_adjs.append(cur_new_trajectory_adjs)
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_localization = [x[:tokenizer_model_max_length] for x in new_localization]
            new_trajectory_adjs = [x[:tokenizer_model_max_length] for x in new_trajectory_adjs]
        batch_size, max_len = len(new_input_embeds), max(x.size(0) for x in new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            size=(batch_size, max_len)
            , fill_value=IGNORE_INDEX
            , dtype=new_labels[0].dtype
            , device=new_labels[0].device
        )
        new_localization_padded = torch.full(
            size=(batch_size, max_len, 2)
            , fill_value=-1
            , dtype=new_localization[0].dtype
            , device=new_localization[0].device
        )
        new_trajectory_adjs_padded = torch.full(
            size=(batch_size, max_len, 3, 100, 2)
            , fill_value=-1
            , dtype=new_trajectory_adjs[0].dtype
            , device=new_trajectory_adjs[0].device
        )
        attention_mask = torch.zeros(size=(batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros(size=(batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for batch_idx, (cur_new_embed, cur_new_labels, cur_new_localization, cur_new_trajectory_adjs) in enumerate(zip(new_input_embeds, new_labels, new_localization, new_trajectory_adjs)):
            cur_len = cur_new_embed.size(0)
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat(
                    tensors=(
                        torch.zeros(
                            size=(max_len - cur_len, cur_new_embed.size(1))
                            , dtype=cur_new_embed.dtype
                            , device=cur_new_embed.device
                        )
                        , cur_new_embed
                    )
                    , dim=0
                ))
                if cur_len > 0:
                    new_labels_padded[batch_idx, -cur_len:] = cur_new_labels
                    new_localization_padded[batch_idx, -cur_len:, ...] = cur_new_localization
                    new_trajectory_adjs_padded[batch_idx, -cur_len:, ...] = cur_new_trajectory_adjs
                    attention_mask[batch_idx, -cur_len:] = True
                    position_ids[batch_idx, -cur_len:] = torch.arange(
                        start=0
                        , end=cur_len
                        , dtype=position_ids.dtype
                        , device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(torch.cat(
                    tensors=(
                        cur_new_embed
                        , torch.zeros(
                            size=(max_len - cur_len, cur_new_embed.size(1))
                            , dtype=cur_new_embed.dtype
                            , device=cur_new_embed.device
                        )
                    )
                    , dim=0
                ))
                if cur_len > 0:
                    new_labels_padded[batch_idx, :cur_len] = cur_new_labels
                    new_localization_padded[batch_idx, :cur_len, ...] = cur_new_localization
                    new_trajectory_adjs_padded[batch_idx, :cur_len, ...] = cur_new_trajectory_adjs
                    attention_mask[batch_idx, :cur_len] = True
                    position_ids[batch_idx, :cur_len] = torch.arange(
                        start=0
                        , end=cur_len
                        , dtype=position_ids.dtype
                        , device=position_ids.device
                    )
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels = None if _labels is None else new_labels_padded
        new_localization = None if _localization is None else new_localization_padded
        new_trajectory_adjs = None if _trajectory_adjs is None else new_trajectory_adjs_padded
        attention_mask = None if _attention_mask is None else attention_mask.to(dtype=_attention_mask.dtype)
        position_ids = None if _position_ids is None else position_ids
        return None, attention_mask, position_ids, past_key_values, new_input_embeds, new_labels, new_localization, new_trajectory_adjs


class PiTeConfig(LlamaConfig):
    model_type = "PiTe"


class PiTeLlamaModel(LlamaModel, PiTeMetaModel):
    config_class = PiTeConfig
    def __init__(self, config):
        super().__init__(config=config)


@dataclass
class PiTeCausalLMOutputWithPast(CausalLMOutputWithPast):
    localization_loss: Optional[torch.FloatTensor] = None
    trajectory_loss: Optional[torch.FloatTensor] = None


class PiTeLlamaForCausalLM(LlamaForCausalLM, PiTeMetaForCausalLM):
    config_class = PiTeConfig
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.model = PiTeLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_model(self) -> nn.Module:
        return self.model
    
    def forward(
        self
        , input_ids: torch.LongTensor = None
        , attention_mask: Optional[torch.Tensor] = None
        , position_ids: Optional[torch.LongTensor] = None
        , past_key_values: Optional[torch.FloatTensor] = None
        , inputs_embeds: Optional[torch.FloatTensor] = None
        , labels: Optional[torch.LongTensor] = None
        , use_cache: Optional[bool] = None
        , output_attentions: Optional[bool] = None
        , output_hidden_states: Optional[bool] = None
        , videos: Optional[torch.FloatTensor] = None
        , localization: Optional[torch.FloatTensor] = None
        , trajectory_adjs: Optional[torch.FloatTensor] = None
        , return_dict: Optional[bool] = True
    ) -> Union[tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, localization, trajectory_adjs = \
                self.prepare_inputs_labels_for_video(
                    input_ids=input_ids
                    , attention_mask=attention_mask
                    , position_ids=position_ids
                    , past_key_values=past_key_values
                    , labels=labels
                    , videos=videos
                    , localization=localization
                    , trajectory_adjs=trajectory_adjs
                )
        outputs = super().forward(
            input_ids=input_ids
            , attention_mask=attention_mask
            , position_ids=position_ids
            , past_key_values=past_key_values
            , inputs_embeds=inputs_embeds
            , labels=labels
            , use_cache=use_cache
            , output_attentions=output_attentions
            , output_hidden_states=True
            , return_dict=return_dict
        )
        loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        localization_loss = None
        if localization is not None:
            if self.config.pretraining_tp > 1:
                localization_projector_slices = self.get_model().localization_projector.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                localization_logits = [F.linear(hidden_states, localization_projector_slices[i]) for i in range(self.config.pretraining_tp)]
                localization_logits = torch.cat(localization_logits, dim=-1)
            else:
                localization_logits = self.get_model().localization_projector(hidden_states)
            localization_logits = localization_logits.view(localization_logits.size(0), -1, 2)
            shift_localization_logits = localization_logits[:, :-1, ...].contiguous()
            shift_localization_labels = localization[:, 1:, ...].contiguous()
            shift_localization_labels = shift_localization_labels.to(shift_localization_logits.device)
            localization_loss = F.l1_loss(shift_localization_logits, shift_localization_labels)
        trajectory_loss = None
        if trajectory_adjs is not None:
            if self.config.pretraining_tp > 1:
                trajectory_projector_slices = self.get_model().trajectory_projector.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                trajectory_logits = [F.linear(hidden_states, trajectory_projector_slices[i]) for i in range(self.config.pretraining_tp)]
                trajectory_logits = torch.cat(trajectory_logits, dim=-1)
            else:
                trajectory_logits = self.get_model().trajectory_projector(hidden_states)
            trajectory_logits = trajectory_logits.view(trajectory_logits.size(0), -1, 3, 100, 2)
            shift_trajectory_logits = trajectory_logits[:, :-1, ...].contiguous()
            shift_trajectory_labels = trajectory_adjs[:, 1:, ...].contiguous()
            shift_trajectory_labels = shift_trajectory_labels.to(shift_trajectory_logits.device)
            trajectory_loss = F.l1_loss(shift_trajectory_logits, shift_trajectory_labels)
        if not return_dict:
            output = (logits, hidden_states)
            return (loss, localization_loss, trajectory_loss) + output if loss is not None else output
        if output_hidden_states:
            return PiTeCausalLMOutputWithPast(
                loss=loss
                , logits=logits
                , localization_loss=localization_loss
                , trajectory_loss=trajectory_loss
                , hidden_states=hidden_states
            )
        else:
            return PiTeCausalLMOutputWithPast(
                loss=loss
                , logits=logits
                , localization_loss=localization_loss
                , trajectory_loss=trajectory_loss
            )
    
    def prepare_inputs_for_generation(
        self
        , input_ids: torch.LongTensor = None
        , past_key_values: Optional[torch.FloatTensor] = None
        , inputs_embeds: Optional[torch.FloatTensor] = None
        , **kwargs
    ) -> dict[str, torch.Tensor]:
        videos = kwargs.pop("videos", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids
            , past_key_values=past_key_values
            , inputs_embeds=inputs_embeds
            , **kwargs
        )
        if videos is not None:
            inputs["videos"] = videos
        return inputs


AutoConfig.register("PiTe", PiTeConfig)
AutoModelForCausalLM.register(PiTeConfig, PiTeLlamaForCausalLM)


if __name__ == "__main__":
    ignore_warnings()
    model_name_or_path = os.path.join(os.getcwd(), "checkpoint", "vicuna-7b-v1.5")
    assert os.path.exists(model_name_or_path)
    model = PiTeLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, device_map="auto")
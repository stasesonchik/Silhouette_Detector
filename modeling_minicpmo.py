# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import types
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from threading import Thread
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import BertTokenizerFast
from transformers import LlamaConfig
from transformers import LlamaModel
from transformers import LogitsWarper
from transformers import PreTrainedModel
from transformers import Qwen2ForCausalLM
from transformers import Qwen2PreTrainedModel
from transformers import TextIteratorStreamer
from transformers import TopKLogitsWarper
from transformers import TopPLogitsWarper
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.modeling_whisper import ACT2FN
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
from transformers.models.whisper.modeling_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from vector_quantize_pytorch import GroupedResidualFSQ
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    _tts_deps = True
except:
    _tts_deps = False

from .configuration_minicpm import ConditionalChatTTSConfig
from .configuration_minicpm import MiniCPMOConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler
from .utils import NumberToTextConverter
from .utils import sentence_end
from .utils import VoiceChecker

logger = logging.getLogger(__name__)


@dataclass
class OmniOutput(ModelOutput):
    text: Optional[Union[str, List[str], Iterator]] = None
    spk_embeds: Optional[torch.FloatTensor] = None
    audio_wav: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None


class MiniCPMOPreTrainedModel(Qwen2PreTrainedModel):
    config_class = MiniCPMOConfig


class MiniCPMO(MiniCPMOPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = Qwen2ForCausalLM(config)
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)  # patch llm

        self.embed_dim = self.llm.config.hidden_size

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            assert _tts_deps, "please make sure vector_quantize_pytorch and vocos are installed."
            self.tts = self.init_tts_module()

        self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.default_tts_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"
        self.force_no_stop = False

        # for stream api
        self.reset_session()

    def reset_session(self):
        self.session_id = None
        self.new_user_msg = True
        self.llm_generated = False
        self.llm_generate_completed = False
        self.llm_past_key_values = None
        self.audio_past_key_values = None  # apm kv cache

    def init_tts(
        self,
        tts_text_tokenizer_path=None,
        vocos_ckpt_path=None,
    ):
        """
        load tts tokenizer and vocos
        1. try load form local 2. try load from huggingface
        """
        from .processing_minicpmo import ChatTTSProcessor

        if tts_text_tokenizer_path is None:
            tts_text_tokenizer_path = os.path.join(self.config._name_or_path, "assets/chattts_tokenizer")
        if not os.path.exists(tts_text_tokenizer_path):
            # try from hf model_id
            tts_text_tokenizer_path = "openbmb/chattts_tokenizer"

        tts_text_tokenizer = BertTokenizerFast.from_pretrained(tts_text_tokenizer_path)
        self.tts_processor = ChatTTSProcessor(text_tokenizer=tts_text_tokenizer)

        if vocos_ckpt_path is None:
            vocos_ckpt_path = os.path.join(self.config._name_or_path, "assets/Vocos.pt")
        if not os.path.exists(vocos_ckpt_path):
            vocos_ckpt_path = hf_hub_download(repo_id="openbmb/MiniCPM-o-2_6", subfolder="assets", filename="Vocos.pt")

        assert os.path.exists(vocos_ckpt_path)
        self.vocos = self.initialize_vocos(vocos_ckpt_path)

    def initialize_vocos(self, ckpt_path):
        feature_extractor = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
                "init_args": {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 100},
            },
        )
        backbone = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.models.VocosBackbone",
                "init_args": {"input_channels": 100, "dim": 512, "intermediate_dim": 1536, "num_layers": 8},
            },
        )
        head = instantiate_class(
            args=(),
            init={"class_path": "vocos.heads.ISTFTHead", "init_args": {"dim": 512, "n_fft": 1024, "hop_length": 256}},
        )
        vocos = Vocos(feature_extractor, backbone, head).to("cuda").eval().to(torch.float32)
        vocos.load_state_dict(torch.load(ckpt_path, weights_only=True, mmap=True))
        return vocos

    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        model = MiniCPMWhisperEncoder(self.config.audio_config)
        return model

    def init_tts_module(self):
        model = ConditionalChatTTS(self.config.tts_config)
        return model

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device

        Returns:
            torch.Tensor: mask

        Examples:
            >>> subsequent_chunk_mask(4, 2)
            [[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vllm_embedding(self, data):
        """
        Compute all visual embeddings, and set into llm embeddings.
        Args:
            data: Dict
                tgt_sizes: image size after patch embedding
                pixel_values: image features
                image_bound: position of each picture corresponding to input_ids
                input_ids: full input_ids, include placeholder
        Returns:
                embedding with vision, vision_hidden_states
        """
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        new_vllm_embedding = vllm_embedding.clone()

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )

                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

        return new_vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(self, data):
        r"""
        Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            audio_outputs = self.apm(wavforms, past_key_values=self.audio_past_key_values, use_cache=True)
            audio_states = audio_outputs.last_hidden_state  # [:, :audio_feat_lengths, :]
            self.audio_past_key_values = audio_outputs.past_key_values

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        dtype = self.apm.embed_positions.weight.dtype
        device = self.apm.embed_positions.weight.device

        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif self.training and dummy:
            dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
            audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return [audio_embeds]

        else:
            return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding
        Returns:
            final embeddings with audio feature
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)
            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if self.config.chunk_input:
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )

                            if embs.shape[0] != len(audio_indices):
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                    f"to input indices of length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)
        elif self.training:
            for i in range(bs):
                # dummy audio_embeddings
                input_embeddings = input_embeddings + audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

        if self.config.init_audio:
            vllm_embedding = self.get_omni_embedding(
                data, input_embeddings=vllm_embedding, chunk_length=self.config.audio_chunk_length
            )

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        # compatible with llama factory
        for key in ["input_ids", "inputs_embeds", "position_ids"]:
            if key in kwargs:
                del kwargs[key]

        return self.llm(input_ids=None, position_ids=position_ids, inputs_embeds=vllm_embedding, **kwargs)

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    def get_sys_prompt(self, ref_audio=None, mode="default", language="zh"):
        """
        Choose different system prompts according to different tasks
        Args:
            ref_audio: if ref_audio is not None, will use the voice cloning prompts, and the voice
                       generated by the model will refer to the timbre of ref audio
            mode:
                "default": default system prompt and not refer to any task
                "omni": input video and audio simultaneously
                "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
                "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
                "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
            language: prompts language, the model has the ability to automatically select the response language
                    based on the question language
        Returns:

        """
        if ref_audio is not None:
            assert isinstance(ref_audio, np.ndarray), "ref_audio error"
        if mode == "omni":
            if language == "zh":
                sys_prompt = "你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。"
                vc_prompt_prefix = sys_prompt + "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                sys_prompt = "You are a helpful assistant. You can accept video, audio and text input and output voice and text. "
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=[],
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        model_output = {}
        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                # if stream return TextIteratorStreamer and output is empty
                outputs = {}
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)

                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs
    
    def get_visual_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states
    
    def generate_visual(self, image_inputs):
        pixel_values = image_inputs['pixel_values']
        tgt_sizes = image_inputs['tgt_sizes']
        model_inputs = {}
        model_inputs["pixel_values"] = pixel_values
        model_inputs["tgt_sizes"] = tgt_sizes
        with torch.inference_mode():
            vision_hidden_states = self.get_visual_embedding(model_inputs)
        return vision_hidden_states
    
    def only_visual_one(self, image):
        sampling = True
        min_new_tokens = 0
        kwargs= {}
        processor = None

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        # processor.image_processor.scale_resolution = 166

        images = [image]
        image_inputs = processor.image_processor(
            images, do_pad=True, max_slice_nums=None, return_tensors="pt"
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        with torch.inference_mode():
            vision_hidden_states = self.generate_visual(image_inputs)

        return vision_hidden_states, image_inputs['pixel_values'], image_inputs['image_sizes'], image_inputs['tgt_sizes']
    
    def chat_one_vision_hidden_state(self, image_features, promt,
        tokenizer=None,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=32768,
        stream=False,
        chunk_input=True,
        omni_input=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        generate_audio=False,
        return_spk_embed=False,
        return_dict=False,
        output_audio_path=None,
        **kwargs,):

        vision_hidden_states = image_features['vision_hidden_states']
        image_sizes = image_features['image_sizes']
        batched = False

        image_inputs = {'image_sizes': image_sizes}

        images_list, msgs_list = [None], [[{'role': 'user', 'content': [None, promt]}]]
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."


        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if c is None:
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                if omni_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=self.default_tts_chat_template if use_tts_template else None,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        inputs = processor.only_text_processor(
            prompts_lists,
            image_inputs,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            chunk_input=chunk_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, outputs = self.generate_text_only(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )
        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            if return_dict:
                return OmniOutput(text=stream_gen())
            else:
                return stream_gen()

        else:
            spk_embeds = wav_numpy = sr = None

            if batched:
                answer = res
            else:
                answer = res[0]

                if use_tts_template and generate_audio:
                    mel_spec = self._generate_mel_spec(inputs, outputs, answer)
                    wav_numpy, sr = self.decode_mel_to_audio(mel_spec, output_audio_path)

            if return_spk_embed:
                spk_embeds = self._get_last_spk_embeds(inputs, outputs)

            if isinstance(answer, list):
                answer = [i.replace(tokenizer.tts_end, "") for i in answer]
            else:
                answer = answer.replace(tokenizer.tts_end, "")

            if return_dict:
                return OmniOutput(text=answer, spk_embeds=spk_embeds, audio_wav=wav_numpy, sampling_rate=sr)
            else:
                return answer

    def generate_text_only(
        self,
        input_ids=None,
        audio_features=[],
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            pass
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        model_output = {}
        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                # if stream return TextIteratorStreamer and output is empty
                outputs = {}
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)

                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=32768,
        stream=False,
        chunk_input=True,
        omni_input=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        generate_audio=False,
        return_spk_embed=False,
        return_dict=False,
        output_audio_path=None,
        **kwargs,
    ):
        """
        Unified chat function

        Args:
            image: use for batch_size=1 vqa, It is not recommended to continue to use this parameter
            msgs: the input chat msgs, support text: (string)  / image: (PIL.Image) / audio (numpy.ndarray)
            tokenizer: tokenizer for llm
            processor: if None, use the default processor
            max_new_tokens: the maximum length of the generation
            min_new_tokens: the minimum length of the generation
            sampling: whether to use sampling decoding or beam search decoding
            max_inp_length: the maximum length of input
            stream: whether to return generator, only used when tts is not required
            chunk_input: whether to split audio into 1s chunks
            omni_input: determine whether it is omni mode
            max_slice_nums: control the maximum number of image slices
            use_image_id: for video understanding or omni understanding, use_image_id should be False
            use_tts_template: if the msgs contain audio, use_tts_template should be True
            generate_audio: whether to generate audio output, only used when return_dict=True
            return_spk_embed: whether to return spk embedding, only used when return_dict=True
            return_dict: whether to return dict
            output_audio_path: audio save path when generate_audio
            **kwargs:
        """
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False

        if generate_audio or return_spk_embed:
            return_dict = True

        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("(<audio>./</audio>)")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                if omni_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=self.default_tts_chat_template if use_tts_template else None,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        inputs = processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            chunk_input=chunk_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            if return_dict:
                return OmniOutput(text=stream_gen())
            else:
                return stream_gen()

        else:
            spk_embeds = wav_numpy = sr = None

            if batched:
                answer = res
            else:
                answer = res[0]

                if use_tts_template and generate_audio:
                    mel_spec = self._generate_mel_spec(inputs, outputs, answer)
                    wav_numpy, sr = self.decode_mel_to_audio(mel_spec, output_audio_path)

            if return_spk_embed:
                spk_embeds = self._get_last_spk_embeds(inputs, outputs)

            if isinstance(answer, list):
                answer = [i.replace(tokenizer.tts_end, "") for i in answer]
            else:
                answer = answer.replace(tokenizer.tts_end, "")

            if return_dict:
                return OmniOutput(text=answer, spk_embeds=spk_embeds, audio_wav=wav_numpy, sampling_rate=sr)
            else:
                return answer

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        tokenizer,
        omni_input=True,
        max_slice_nums=None,
        ls_temperature=1.0,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream, Only support batch_size=1
        Args:
            session_id: Note: new connection should use a new session_id
        """
        assert session_id is not None
        if self.session_id is None or session_id != self.session_id:  # new session
            self.is_first = True
        else:
            self.is_first = False

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):  # audio
                audios.append(c)
                cur_msgs.append("(<audio>./</audio>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error("Invalid content type:", c)

        cur_contents = "".join(cur_msgs) if omni_input else "\n".join(omni_input)
        if not self.is_first and self.new_user_msg and msg["role"] == "user":  # new user add im_start
            if self.llm_generated:
                if self.llm_generate_completed:
                    msg["content"] = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                else:  # break llm gen, add tts_eos
                    msg["content"] = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
            else:
                msg["content"] = "<|im_start|>user\n" + cur_contents
            self.new_user_msg = False
        else:
            msg["content"] = cur_contents

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None  # apm kv cache

        if self.is_first:
            # init pask_key_values
            logger.info(f"new session_id: {session_id}, reset kv cache")
            self.reset_session()
            self.session_id = session_id

            prompt = tokenizer.apply_chat_template(
                copy_msgs, tokenize=False, add_generation_prompt=False, chat_template=self.default_tts_chat_template
            )
            add_special_tokens = True  # add bos
        else:
            prompt = copy_msgs[0]["content"]
            add_special_tokens = False

        model_inputs = self.processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
        ).to(self.device)

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=True
        )

        if self.is_first:
            # clean audio_past_key_values after first prefill
            self.audio_past_key_values = None

        if self.llm_past_key_values is not None:
            cache_length = self.llm_past_key_values[0][0].shape[2]
        else:
            cache_length = 0

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill and predict listen/speak label
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,  # position_ids,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = outputs["past_key_values"]
        return

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        tokenizer,
        max_new_tokens=512,
        min_new_tokens=0,
        sampling=True,
        generate_audio=True,
        enable_regenerate=False,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream
        Args:
        """
        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        # do generate
        # reset buffer
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None  # apm kv cache

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        generate_prompt = "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"
        input_ids = tokenizer(generate_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

        spk_start_idx = torch.where(input_ids[0] == tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids[0] == tokenizer.spk_end_id)[0]
        spk_bounds = [
            torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
        ]  # List[Tensor], (1,2)

        cache_length = past_length = self.llm_past_key_values[0][0].shape[2]
        attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

        generation_config["max_new_tokens"] = max_new_tokens
        streamer = self.llm_generate_chunk(input_ids, attention_mask, tokenizer, terminators, generation_config)

        if generate_audio:
            result = self._generate_mel_spec_audio_streaming(
                spk_bounds, streamer, output_chunk_size=25, enable_regenerate=enable_regenerate
            )
            return result
        else:
            return streamer

    def llm_generate_chunk(self, input_ids, attention_mask, tokenizer, terminators, generation_config):
        def check_uncompleted_token(ids):
            cur_text = tokenizer.decode(ids)
            end = len(ids)
            while cur_text[-1] == "�":
                end -= 1
                if end == 0:
                    break
                cur_text = tokenizer.decode(ids[:end])
            return end

        max_new_tokens = int(generation_config.pop("max_new_tokens", 2048))
        new_len = 0
        first_chunk = True
        eos = False
        left_ids = None

        while True:
            outputs = self.llm.generate(
                input_ids=input_ids,
                past_key_values=self.llm_past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                max_new_tokens=3,  # reduce first token delay
                pad_token_id=0,
                output_hidden_states=True if first_chunk else False,
                return_dict_in_generate=True,
                eos_token_id=terminators,
                **generation_config,
            )
            if outputs.sequences[0, -1] in terminators:
                eos = True
            input_len = input_ids.shape[1]
            cur_ids = outputs.sequences[:, input_len:]
            new_len += cur_ids.shape[1]

            if left_ids is not None and left_ids.shape[1] > 0:
                cur_ids = torch.cat([left_ids, cur_ids], dim=1)
            end = check_uncompleted_token(cur_ids[0])
            left_ids = cur_ids[:, end:]
            cur_ids = cur_ids[:, :end]
            text = self._decode_text(cur_ids, tokenizer)[0] if end > 0 else ""

            self.llm_past_key_values = outputs.past_key_values
            input_ids = outputs.sequences[:, -1:]
            cache_length = past_length = self.llm_past_key_values[0][0].shape[2]
            attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

            res = {"text": text}
            if first_chunk:
                res["hidden_states"] = outputs.hidden_states
                first_chunk = False
            yield res

            if eos:
                self.llm_generate_completed = True
                break
            if new_len >= max_new_tokens:
                logger.debug(f"LLM generation {new_len} exceeds max_new_tokens({max_new_tokens}), break.")
                break

    def prepare_tts_text(self, text):
        tts_tokens = self.tts_processor.text_tokenizer.encode(text, add_special_tokens=False)
        tts_tokens_len = len(tts_tokens)
        if tts_tokens_len < self.tts.streaming_text_reserved_len:
            num_pad_tokens = self.tts.streaming_text_reserved_len - tts_tokens_len

            pad_str = "[Etts]" + "[PAD]" * (num_pad_tokens - 1)
        else:
            tts_tokens = tts_tokens[0 : self.tts.streaming_text_reserved_len]
            tts_tokens_len = len(tts_tokens)
            text = self.tts_processor.text_tokenizer.decode(tts_tokens, add_special_tokens=False)
            pad_str = ""
        spk_emb_placeholder_tts = "[spk_emb]" * self.tts.num_spk_embs

        new_text_tts = f"[Stts]{spk_emb_placeholder_tts}{text}{pad_str}[Ptts]"
        return new_text_tts, tts_tokens_len

    def get_tts_text_start_token_ids(self):
        text = "[Stts]" + "[spk_emb]" * self.tts.num_spk_embs
        tts_input_ids = self.tts_processor.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].cuda()
        return tts_input_ids

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = (
            1 + self.tts.num_spk_embs * self.tts.use_speaker_embedding + self.tts.streaming_text_reserved_len + 1
        )
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _get_last_spk_embeds(self, inputs, outputs):
        last_hidden_states = [hs[-1] for hs in outputs.hidden_states]

        # batch = 1
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])

        # last spk
        spk_bound = inputs["spk_bounds"][0][-1]

        spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]
        return spk_embeds

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to("cuda", dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626, top_P=self.tts.top_p, top_K=self.tts.top_k, repetition_penalty=self.tts.repetition_penalty
        )

        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
            else:
                begin = (
                    chunk_idx * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids, position_ids=position_ids, past_key_values=past_key_values
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=self.force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

        mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids)
        return mel_spec

    def _linear_overlap_add2_wav(self, frames: List[torch.Tensor], overlap: int):
        """
        Merge two audio waveforms with smooth in streaming audio generation.
        Borrowed some codes from `https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py`
        """
        assert len(frames) == 2
        device = frames[0].device
        dtype = frames[0].dtype
        # shape = frames[0].shape[:-1]

        frame0_length = frames[0].shape[-1]
        frame1_length = frames[1].shape[-1]
        total_size = frame0_length + frame1_length - overlap
        weight_len = max(frame0_length, frame1_length) + overlap
        t = torch.linspace(0, 1, weight_len + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (t - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(total_size, device=device, dtype=dtype)
        offset: int = 0

        out[offset : offset + frame0_length] += weight[-frame0_length:] * frames[0]
        sum_weight[offset : offset + frame0_length] += weight[-frame0_length:]
        offset += frame0_length - overlap
        out[offset : offset + frame1_length] += weight[:frame1_length] * frames[1]
        sum_weight[offset : offset + frame1_length] += weight[:frame1_length]

        assert sum_weight.min() > 0
        out = out / sum_weight
        return out[:frame0_length], out[frame0_length:]

    def _generate_mel_spec_audio_streaming(
        self,
        spk_bounds,
        streamer,
        output_chunk_size=25,
        spk_embeds=None,
        prev_seg_text_ids=None,
        prev_seg_text_left="",
        prev_seg_audio_ids=None,
        enable_regenerate=False,
    ):
        # get spk_embedding
        gen_text = ""
        tts_text = ""
        new_segment_gen = False
        if spk_embeds is None:
            spk_bound = spk_bounds[0][-1]
            r = next(streamer)
            txt = r["text"]
            gen_text += txt.split("<|tts_eos|>")[0]
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
            last_hidden_states = r["hidden_states"][0][-1][0]  # output: (input_seq_len, dim)
            spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]

        # init past_key_values
        logits_warpers, logits_processors = gen_logits(
            num_code=626, top_P=self.tts.top_p, top_K=self.tts.top_k, repetition_penalty=self.tts.repetition_penalty
        )
        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )
        tts_start_token_len = 1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs
        dtype = self.tts.emb_text.weight.dtype
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]
        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        # prefill prev segment for smooth
        chunk_idx = 0
        new_ids_len = 0
        prev_text_len = 0
        if prev_seg_text_ids is not None and prev_seg_audio_ids is not None:
            tts_token_lens = prev_seg_text_ids.shape[1]
            # assert tts_token_lens % self.tts.streaming_text_chunk_size == 0
            streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
            position_ids = torch.arange(
                0, tts_token_lens + tts_start_token_len, dtype=torch.long, device=self.tts.device
            ).unsqueeze(0)

            text_input_ids = self.get_tts_text_start_token_ids()
            text_input_ids = torch.cat([text_input_ids, prev_seg_text_ids], dim=1)
            past_key_values = self.tts.prefill_text(
                input_ids=text_input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                lm_spk_emb_last_hidden_states=spk_embeds,
            )
            past_key_values = self.tts.prefill_audio_ids(
                input_ids=prev_seg_audio_ids[:, :-1, :],
                # not prefill last id, which will be input_id of next generation
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
            )

            # update init
            chunk_idx += int(tts_token_lens / self.tts.streaming_text_chunk_size)
            audio_input_ids = torch.cat([audio_input_ids, prev_seg_audio_ids], dim=1)
            text = self.tts_processor.text_tokenizer.decode(prev_seg_text_ids[0].tolist(), add_special_tokens=False)

            gen_text += text
            gen_text += prev_seg_text_left
            prev_text_len = len(gen_text)  # takecare the position
            new_ids_len += prev_seg_audio_ids.shape[1]

        prev_wav = None
        eos_lab = False
        stop = False
        shift_len = 180
        voice_checker = VoiceChecker()
        number_converter = NumberToTextConverter()
        lang = None
        gen_text_raw = gen_text
        for t, r in enumerate(streamer):
            t += 1
            txt = r["text"]
            txt = txt.split("<|tts_eos|>")[0]
            gen_text_raw += txt
            if t == 1 and txt == "" and prev_seg_text_ids is not None:
                logger.warning("New segment is empty, generation finished.")
                return
            if t <= 2:  # do just one time, more token greater certainty
                lang = number_converter.detect_language(gen_text_raw)
            gen_text += number_converter.replace_numbers_with_text(txt, lang).replace("*", "")  # markdown **

            # TODO speed up
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)

            if tts_token_lens >= self.tts.streaming_text_reserved_len - shift_len:
                end_c = sentence_end(txt)
                if end_c:
                    end_c_idx = gen_text.rfind(end_c)
                    assert end_c_idx != -1
                    text_left = gen_text[end_c_idx + 1 :]
                    gen_text = gen_text[: end_c_idx + 1]
                    tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
                    new_segment_gen = True
                    logger.debug(
                        f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, starting a new segment generation"
                    )
                    break

            if tts_token_lens >= (chunk_idx + 1) * self.tts.streaming_text_chunk_size:

                # do prefill and generate
                if chunk_idx == 0:
                    begin = 0
                    end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len
                else:
                    begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
                    end = min(
                        (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len, condition_length - 1
                    )

                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = (
                    outputs.audio_input_ids
                )  # [1,seq_len,4] seq_len=tts.streaming_text_reserved_len + 3 + len(new_ids)
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)  # [1,100,50] -> [50*256]

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)

                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)

                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    eos_lab = True
                    break

        if not eos_lab and tts_text:
            logger.debug("eos_lab False, Generation continue.")

            if chunk_idx == 0:
                begin = 0
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
            end = tts_token_lens + tts_start_token_len + 1  # 1 for [Etts]
            if end > begin:
                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )

            while True:
                # temp = [0.1, 0.3, 0.1, 0.3] if chunk_idx < 21 else [0.1] * self.tts.num_vq
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    # temperature=torch.tensor([0.1] * self.tts.num_vq, dtype=torch.float, device=self.tts.device),
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)
                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)
                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > 2048:
                    stop = True
                    logger.debug("Generation length > 2048, stopped.")
                    break

        if prev_wav is not None:
            cur_text = gen_text_raw[prev_text_len:]
            yield OmniOutput(text=cur_text, audio_wav=prev_wav, sampling_rate=sr)  # yield last chunk wav without smooth

        if new_segment_gen and not stop:
            logger.debug(
                f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, start a new segment generation"
            )
            tid_len = 5  # self.tts.streaming_text_chunk_size
            prev_seg_text_ids = tts_input_ids[:, end - 1 - tid_len : end - 1]  # exclude last Etts
            aid_len = 50  # int(tid_len * new_ids_len / tts_token_lens)
            prev_seg_audio_ids = outputs.new_ids[:, -aid_len:, :]

            result = self._generate_mel_spec_audio_streaming(
                spk_bounds,
                streamer,
                output_chunk_size,
                spk_embeds,
                prev_seg_text_ids,
                text_left,
                prev_seg_audio_ids,
                enable_regenerate=enable_regenerate,
            )
            for res in result:
                yield res

    def decode_mel_to_audio(self, mel_spec, output_path=""):
        with torch.inference_mode():
            wav_numpy = self.vocos.decode(mel_spec.float()).cpu().squeeze()
            sr = 24000
        if output_path:
            sf.write(output_path, wav_numpy.numpy(), samplerate=sr)
            logger.info(f"Audio saved to {output_path}")
        return wav_numpy, sr


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference
class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, embed_dim)`):
                Hidden states to be fed into the encoder layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, tgt_len, src_len)`):
                Attention mask where padding elements are indicated by large negative values.
            layer_head_mask (`torch.FloatTensor` of shape `(encoder_attention_heads,)`):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention weights.
            past_key_values (`EncoderDecoderCache`, *optional*):
                Past key-value pairs used for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether or not to return updated `past_key_values` for caching.

        Returns:
            A tuple of shape `(hidden_states, optional(attn_weights), optional(past_key_values))`.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):
        r"""
        Forward pass of the Whisper encoder.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of log-mel features extracted from the raw audio waveform. Typically generated
                by a feature extractor (e.g., `WhisperFeatureExtractor`) that processes `.flac` or `.wav`
                files into padded 2D mel spectrogram frames. These features are projected via convolution layers
                (`conv1` and `conv2`) and then transformed into embeddings for the encoder.

            attention_mask (`torch.Tensor`, *optional*):
                Not used by Whisper for masking `input_features`, but included for API compatibility with
                other models. If provided, it is simply ignored within the model. By default, Whisper
                effectively ignores silence in the input log-mel spectrogram.

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected attention heads. The elements should be either 1 or 0, where:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked** (i.e., the attention head is dropped).

            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all encoder layers. If set to `True`, the
                returned tuple (or `BaseModelOutputWithPast`) will contain an additional element with
                attention weights for each encoder layer.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. If set to `True`, the returned
                tuple (or `BaseModelOutputWithPast`) will contain a tuple of hidden states, including the
                initial embedding output as well as the outputs of each layer.

            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` (a subclass of `ModelOutput`) instead
                of a plain tuple. If set to `True`, the output will be a `BaseModelOutputWithPast` object,
                otherwise it will be a tuple.

            past_key_values (`EncoderDecoderCache`, *optional*):
                When using caching for faster inference, this is an object that stores the key-value pairs
                for attention states. If provided, the model will append new states to the existing cache
                and return the updated cache. This speeds up sequential decoding or chunked inference.

                - If `past_key_values` is `None`, no past states are used or returned.
                - If `past_key_values` is not `None` and `use_cache=True`, the model will use the provided
                cache and return the updated cache (as `next_encoder_cache`).

            use_cache (`bool`, *optional*):
                Whether or not the model should use caching (`past_key_values`) to speed up processing
                during inference. When set to `True`, the model will:
                - Inspect and use `past_key_values` if provided.
                - Return updated `past_key_values` (under the name `next_encoder_cache` in
                    `BaseModelOutputWithPast`).

        Returns:
            `BaseModelOutputWithPast` or `tuple` (depending on `return_dict`):
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, which contains:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The output of the final encoder layer.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True`):
                Hidden states of the model at each layer (including the initial projection).
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True`):
                Attention weights from each encoder layer.
                - **past_key_values** (an object of type `EncoderDecoderCache` or `None`, *optional*):
                Updated cache of key-value pairs if `use_cache=True`.

                If `return_dict=False`, a tuple is returned, where the format is:
                `(last_hidden_state, hidden_states, attentions)`, with `hidden_states` and `attentions`
                only present if their respective `output_*` arguments are set to `True`.

        Example:
            >>> from transformers import AutoFeatureExtractor, WhisperConfig, WhisperForConditionalGeneration
            >>> import torch

            >>> # Load a feature extractor and a Whisper model
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
            >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

            >>> # Assume you have audio (list of floats or numpy array) loaded from a file
            >>> # Then extract the mel features:
            >>> input_features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features

            >>> # Forward pass
            >>> outputs = model.encoder(
            ...     input_features=input_features,
            ...     output_hidden_states=True,
            ...     output_attentions=True,
            ...     use_cache=True
            ... )

            >>> # Retrieve the last hidden state
            >>> last_hidden_state = outputs.last_hidden_state
            >>> print(last_hidden_state.shape)
            torch.Size([batch_size, seq_length, hidden_size])

            >>> # Retrieve the intermediate hidden states if output_hidden_states=True
            >>> all_encoder_hidden_states = outputs.hidden_states

            >>> # Retrieve attention weights if output_attentions=True
            >>> all_encoder_attentions = outputs.attentions

            >>> # Retrieve updated past key values if use_cache=True
            >>> encoder_cache = outputs.past_key_values
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.coef = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        del y
        y = self.pwconv2(x)
        del x
        if self.coef is not None:
            y *= self.coef
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)

        x = y + residual
        del y

        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class GFSQ(nn.Module):
    def __init__(
        self,
        dim: int,
        levels: List[int],
        G: int,
        R: int,
        eps=1e-5,
        transpose=True,
    ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=list(levels),
            num_quantizers=R,
            groups=G,
        )
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2) if self.transpose else feat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x.transpose_(1, 2)
        _, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        return ind.transpose_(1, 2) if self.transpose else ind


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAEDecoder(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer=12,
        bn_dim=64,
        hidden=256,
        kernel=7,
        dilation=2,
        up=False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden,
                    hidden * 4,
                    kernel,
                    dilation,
                )
                for _ in range(n_layer)
            ]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in(x)
        del x
        for f in self.decoder_block:
            y = f(y, conditioning)

        x = self.conv_out(y)
        del y
        return x


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/dvae.py`
class DVAE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        coef = torch.rand(100)
        self.coef = nn.Parameter(coef.unsqueeze(0).unsqueeze_(2))

        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, 512, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.GELU(),
        )

        self.encoder = DVAEDecoder(
            idim=512,
            odim=1024,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.decoder = DVAEDecoder(
            idim=512,
            odim=512,
            hidden=256,
            n_layer=12,
            bn_dim=128,
        )

        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)

        self.vq_layer = GFSQ(
            dim=1024,
            levels=(5, 5, 5, 5),
            G=2,
            R=2,
        )

    @torch.inference_mode()
    def forward(self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode") -> torch.Tensor:
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:
            mel = inp.clone()
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),
            ).unsqueeze_(0)
            del mel
            x = self.encoder(x)
            ind = self.vq_layer(x)
            del x
            return ind

        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp

        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )

        dec_out = self.out_conv(
            self.decoder(
                x=vq_feats,
            ),
        )

        del vq_feats

        return torch.mul(dec_out, self.coef, out=dec_out)


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """
    Replace consecutive `num_spk_embs` speaker embedding placeholders in input_embeds with pre-prepared speaker embeddings. This is an in-place replacement, no new tensor is created, so no value is returned.

    Args:
        input_ids (torch.Tensor): Input ID tensor, shape [batch_size, seq_len_max]
        spk_emb (torch.Tensor): Speaker embedding tensor, shape [batch_size, num_spk_emb, hidden_dim]
        input_embeds (torch.Tensor): Input embedding tensor, shape [batch_size, seq_len_max, hidden_dim]
        spk_emb_token_id (int): ID of the speaker embedding token
        num_spk_embs (int): Number of speaker embeddings

    Returns:
        None
    """

    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        input_ids_ = input_ids[idx]  # [seq_len_max]
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

    return


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_chunk_length (int, optional): Length of each streaming chunk. Defaults to 50.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`
    causal_mask = torch.full((1, past_seen_tokens + inputs_embeds.shape[1]), fill_value=0, dtype=dtype, device=device)

    # Calculate the start of invisible text tokens
    invisible_text_tokens_start = (
        min(
            math.ceil((past_seen_tokens - streaming_reserved_length) / streaming_audio_chunk_size)
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)

    # Set invisible text tokens to min_dtype (effectively -inf)
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask
    causal_mask[0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1].masked_fill_(
        streaming_tts_text_mask == 0, min_dtype
    )

    # Add extra dimensions for batch and heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`
class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    finished: bool = None


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ConditionalChatTTS(PreTrainedModel):
    """A conditional text-to-speech model that can generate speech from text with speaker conditioning.

    This model extends PreTrainedModel to provide text-to-speech capabilities with:
    - LLM hidden state conditioning
    - Streaming generation

    The model uses a transformer architecture with LLM hidden states and can operate in both
    streaming and non-streaming modes for flexible deployment.

    The model process sequence in the following format:
    | text bos token | LLM embedding projected to tts embedding space | text tokens (fixed length, reserved for future tokens) | audio bos token | audio tokens (audio token length is not fixed)| audio eos token |

    The format is designed to support LLM-conditioned streaming audio generation.

    Usage:
    To support streaming generation, two global variables should be maintained outside of the model.
        1. `audio_input_ids`: stores *discrete* audio codes. It is a tensor with shape [1, sequence length+1, num_vq].
        2. `past_key_values`: stores the KV cache for both text tokens and audio codes. It is a list of tuples, each tuple contains two tensors with shape [1, num_attention_heads, sequence length, hidden_size // num_attention_heads]

    where `num_vq` is the number of audio codebooks, in default setting, it is `4`.

    1. Create an empty `past_key_values` with
    ```python
    initial_kv_cache_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len # where `1` denotes the `bos` token
    dtype = model.emb_text.weight.dtype
    device = model.emb_text.weight.device
    past_key_values = [
        (
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device),
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device)
        )
        for _ in range(model.config.num_hidden_layers)
    ]

    2. At the same time, create an empty `audio_input_ids` with shape [1, sequence length, num_vq], `num_vq` denotes multiple layer audio codebooks. But here we also include text tokens in the sequence, but they will be zeros, and will not be used, just a placeholder.

    ```python
    initial_audio_input_ids_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len + 1
    # [bos token, speaker embeddings, text tokens, audio bos token]
    audio_input_ids = torch.zeros(batch_size=1, initial_audio_input_ids_length, model.num_vq)
    ```

    2. Prefill some text tokens to TTS model (for example, 10 tokens) using `prefill_text` method.

    ```python
    outputs = llm.generate(**kwargs)
    llm_tokens = some_function_to_extract_llm_tokens(outputs)
    lm_spk_emb_last_hidden_states = some_function_to_extract_lm_spk_emb_last_hidden_states(outputs)
    tts_text_input_ids = tts_tokenizer.encode(llm_tokenizer.decode(llm_tokens))
    # here assume we are prefilling text token 0 to text token 9 (included), totally 10 tokens.
    begin = 0
    end = 9+1
    position_ids = torch.arange(begin, end, dtype=torch.long, device=device)

    past_key_values = model.prefill_text(
        input_ids=tts_text_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
    )
    ```

    3. Make a `streaming_tts_text_mask` to denote which position contains valid text tokens, similar to `attention_mask` in standard causal attention.

    ```python
    streaming_tts_text_mask = torch.zeros(model.streaming_reserved_length)
    streaming_tts_text_mask[0:end] = 1 # denotes these post
    ```

    3. Generate audio codes using `generate` method.

    ```python
    outputs = model.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=50,
    )

    # update past_key_values and input_ids
    past_key_values = outputs.past_key_values
    audio_input_ids = outputs.input_ids
    ```

    The `past_key_values` is extended by `max_new_token=50`, and `audio_input_ids` is also extended by `max_new_token=50` after `generate` calling.

    4. Notice that after prefilling `10` text tokens, the model can generate up to `50` audio tokens, if you want to generate more audio tokens, you need to prefill next `10` text tokens. And it is okay to only generate `25` audio tokens for faster initial response.

    5. Repeat steps `2,3,4` as needed in your streaming audio generation cases, but ensure usage complies with the following guidelines discussed above.
    """

    config_class = ConditionalChatTTSConfig

    def __init__(self, config: ConditionalChatTTSConfig):
        super().__init__(config)

        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_size = config.streaming_text_chunk_size
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        if self.config.use_mlp:
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )
        dvae = DVAE()
        self.dvae = dvae

        model_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
        )

        model = LlamaModel(model_config)
        self.model = model

    @torch.inference_mode()
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Merge `input_ids` and `lm_spk_emb_last_hidden_states` to `inputs_embeds`.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            lm_spk_emb_last_hidden_states (Optional[torch.Tensor], optional): Last hidden states of speaker embeddings from the language model. Defaults to None.

        Raises:
            NotImplementedError: If speaker embedding is not used and language model hidden states are not implemented.

        Returns:
            torch.Tensor: Prepared input embeddings for the model.
        """
        assert input_ids.shape[0] == 1

        # Embed input_ids to input_embeds
        inputs_embeds = self.emb_text(input_ids)

        # Inject speaker embedding to input_embeds if it exists
        if self.use_speaker_embedding:
            spk_emb_mask = input_ids == self.spk_emb_token_id
            if spk_emb_mask.any():
                assert lm_spk_emb_last_hidden_states is not None
                # Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim] -> [batch_size, num_spk_emb, self.hidden_size]
                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(self.projector.linear1.weight.dtype)
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )
        else:
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Prefill a chunk of new text tokens in streaming setting.
        Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

        Args:
            input_ids (Tensor): Tensor of shape [batch_size, seq_len]
            position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
            past_key_values (List[Tuple[Tensor]]): KV Cache of all layers, each layer is a tuple (Tensor, Tensor) denoting keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
            lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.
            lm_last_hidden_states (Tensor, optional): _description_. Defaults to None.

        Note that all `batch_size` should be `1`.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # Merge text and LLM embeddings
        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

        # Clone KV Cache
        past_key_values_for_prefill = []
        for i in range(len(past_key_values)):
            past_key_values_for_prefill.append(
                (
                    past_key_values[i][0][:, :, : position_ids[:, 0], :].clone(),
                    past_key_values[i][1][:, :, : position_ids[:, 0], :].clone(),
                )
            )

        # Model forward
        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,  # because for text, it is standard causal attention mask, do nothing
            position_ids=position_ids,  # position_ids denotes the position of new text tokens in the sequence
            past_key_values=past_key_values_for_prefill,  # `past_key_values` will be updated by the model
            inputs_embeds=inputs_embeds,  # contains text and language model embedding
            use_cache=True,
            output_attentions=False,
            cache_position=position_ids,  # which new positions will use this cache, basically the same as position_ids
        )

        # Get model updated KV Cache
        past_key_values_for_prefill_updated = outputs_prefill.past_key_values

        # Update generated KV Cache to input `past_key_values`
        for layer_idx in range(len(past_key_values)):
            # Update keys
            past_key_values[layer_idx][0][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                past_key_values_for_prefill_updated[layer_idx][0][
                    :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                ].clone()
            )
            # Update values
            past_key_values[layer_idx][1][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                past_key_values_for_prefill_updated[layer_idx][1][
                    :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                ].clone()
            )

        # TODO: del past_key_values_for_prefill_updated recursively
        # TODO: del outputs_prefill recursively

        return past_key_values

    @torch.inference_mode()
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):
        """Prefill a chunk of audio ids to the model. Used in sliding-window long audio generation.
        Specifically, prefill many audio ids (typically from last window) to the model in the new window.

        Args:
            input_ids (torch.Tensor): (1, seq_len, num_vq) Audio input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)  # [1,seq_len,768]
        input_len = input_ids.shape[1]

        if add_audio_bos:
            narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)
            input_len += 1

        past_key_values_length = past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_key_values_length, past_key_values_length + input_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        cache_position = position_ids.clone()
        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values[0][0].shape[2],
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )  # [1, 1, 1, past_key_values_length + input_len]

        # Model forward
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            cache_position=cache_position,
        )
        past_key_values = outputs.past_key_values
        return past_key_values

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: List[LogitsWarper] = [],
        logits_processors: List[CustomRepetitionPenaltyLogitsProcessorRepeat] = [],
        show_tqdm=False,
    ):
        """Generate audio codes in streaming setting or non-streaming setting.
        Specifically speaking, generate audio codes when not all text tokens are prefilled.

        Always pass a valid `past_key_values` to the method. The method does not do `prefill` by itself. It relies on `prefill_text` method to provide valid `past_key_values`. Please refer to docstring of this class for more details.

        In this method, we borrowed a lot of codes from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/gpt.py`.

        Args:
            input_ids (torch.Tensor): Input token ids.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): Past key values for attention mechanism.
            temperature (torch.Tensor): Temperature for sampling.
            eos_token (Union[int, torch.Tensor]): End of sequence token.
            streaming_tts_text_mask (Optional[torch.Tensor], optional): Mask for streaming TTS text. Defaults to None.
            max_new_token (int, optional): Maximum number of new tokens to generate. Defaults to 50.
            logits_warpers (List[LogitsWarper], optional): List of logits warpers. Defaults to [].
            logits_processors (List[CustomRepetitionPenaltyLogitsProcessorRepeat], optional): List of logits processors. Defaults to [].
            show_tqdm (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            GenerationOutputs: Generation outputs.
        """

        # We only support batch size `1` for now
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # fix: this should not be `input_ids.shape[1]`
        # start_idx = input_ids.shape[1]
        start_idx = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()

        temperature = temperature.unsqueeze(0).expand(input_ids.shape[0], -1).contiguous().view(-1, 1)

        progress = input_ids.shape[1]

        # Pre-allocate input_ids, shape is [batch_size=1, max_possible_seq_len, self.num_vqs]
        input_ids_buf = torch.zeros(
            input_ids.shape[0],  # batch_size
            progress + max_new_token,  # max_possible_seq_len = input_ids.shape[1] + max_new_token
            input_ids.shape[2],  # self.num_vqs
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        # Copy existing `input_ids` to `input_ids_buf`
        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)

        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        condition_length = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

        for i in range(max_new_token):
            # Prepare generation inputs
            audio_bos = False

            # If this is the first audio token, the case is SPECIAL
            if progress == condition_length:
                audio_bos = True

            assert progress == (
                past_key_values[0][0].shape[2] + 1
            )  # If you are using according to the guidelines, this should be passed.

            if audio_bos:
                # Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict a new audio token. This is a special case because without the `audio bos token`, it is impossible to generate the first audio token in our streaming setting.
                narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
                inputs_embeds = self.emb_text(narrowed_input_ids)
                del narrowed_input_ids
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`.
                narrowed_input_ids = input_ids.narrow(dim=1, start=input_ids.shape[1] - 1, length=1)
                code_emb = [self.emb_code[i](narrowed_input_ids[:, :, i]) for i in range(self.num_vq)]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.tensor(
                [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
            ).unsqueeze(0)

            cache_position = position_ids.clone()

            # Make causal mask
            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_key_values[0][0].shape[2],
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )

            # Model forward
            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                cache_position=cache_position,
            )

            del position_ids
            del inputs_embeds
            del cache_position
            del causal_mask

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            # logits = logits[:, -1].float()
            logits = logits.narrow(1, -1, 1).squeeze_(1).float()

            # logits = rearrange(logits, "b c n -> (b n) c")
            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))
            # logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
            input_ids_sliced = input_ids.narrow(
                1,
                start_idx,
                input_ids.size(1) - start_idx,
            ).permute(0, 2, 1)
            logits_token = input_ids_sliced.reshape(
                input_ids_sliced.size(0) * input_ids_sliced.size(1),
                -1,
            ).to(self.device)
            del input_ids_sliced

            logits /= temperature

            if not audio_bos:
                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)
            if not audio_bos:
                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

            del logits_token

            if i < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1)  # .to(finish.device)

            del scores

            # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            # Store new `token` into `input_ids_buf`
            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

            if i == 0 and finish.any():
                # raise Exception
                break

            del idx_next
            progress += 1
            input_ids = input_ids_buf.narrow(1, 0, progress)

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            if show_tqdm:
                logger.info(f"incomplete result. hit max_new_token: {max_new_token}")

        del input_ids_buf

        if finish.all():
            # the last may contains eos token
            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:
            # there is no eos token
            genrated_input_ids = input_ids[:, condition_length:, :]

        return ConditionalChatTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=input_ids,  # for update purpose
            past_key_values=past_key_values,  # for update purpose
            finished=finish.all(),
        )

    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):
        """Decode discrete audio codes to mel spectrograms.

        Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/core.py`

        Args:
            result_list (List[torch.Tensor]): Audio codes output from `generate`.

        Returns:
            torch.Tensor: Mel spectrograms.
        """

        decoder = self.dvae
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device,
        )
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
            del src

        mel_specs = decoder(batch_result)
        del batch_result
        return mel_specs


# Borrowed from `https://github.com/2noise/ChatTTS/blob/main/ChatTTS/model/processors.py`
def gen_logits(
    num_code: int,
    top_P=0.7,
    top_K=20,
    repetition_penalty=1.0,
):
    logits_warpers = []
    if top_P is not None:
        logits_warpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        logits_warpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors


# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            # This clo≠clo≠clone call is needed to avoid recapturing cuda graphs with →rch.comπ≤→rch.comπ≤torch.compile's  mode=reduce−overheadmode=reduce-overheadmode="reduce-overhead, as otherwise the input positionidspositionidsposition_ids would have various stride during the decoding. Here, simply using .contiguous().contiguous().contiguous() is not sufficient as in the batch size = 1 case, positionidspositionidsposition_ids is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    # if ∈putsembeds∈putsembedsinputs_embeds are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        # The clone here is for the same reason as for positionidspositionidsposition_ids.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            # "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

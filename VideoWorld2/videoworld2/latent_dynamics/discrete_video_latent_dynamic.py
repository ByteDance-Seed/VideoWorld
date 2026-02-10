# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The network definition for discrete video tokenizer with VQ, LFQ, FSQ or ResidualFSQ. """
from collections import OrderedDict, namedtuple
from transformers import LlamaConfig, LlamaModel
import torch
from loguru import logger as logging
from torch import nn
import copy
from videoworld2.latent_dynamics.modules import Decoder3DType, DiscreteQuantizer, Encoder3DType
from videoworld2.latent_dynamics.modules.layers3d import CausalConv3d, LDMDecoderFactorized, LDMEncoderFactorized, CausalResnetBlockFactorized3d
from videoworld2.latent_dynamics.modules.quantizers import InvQuantizerJit


NetworkEval = namedtuple("NetworkEval", ["quant_info", "quant_loss", "quant_codes", "ldm_query", "reconstructions", "mid_h"])

class CustomEncoderJIT(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.encoder = original_model.encoder
        self.quant_conv = original_model.quant_conv
        self.quantizer = original_model.quantizer
    
    def forward(self, x):
        h, query = self.encoder(x)
        
        query = self.quant_conv(query) #B, 16, T-1, 1, 1
        quant_query = self.quantizer(query)
        quant_info, quant_codes, quant_loss = quant_query

        return h, quant_info, (quant_codes, quant_loss), query

class CustomDecoderJIT(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.inv_quant = InvQuantizerJit(original_model.quantizer)
        self.post_quant_conv = original_model.post_quant_conv
        self.decoder = original_model.decoder
    
    def forward(self, encoded_video, code_b):
        quant_b = self.inv_quant(code_b)
        
        quant_b = self.post_quant_conv(quant_b) #B, 16, T-1, 1, 1
        reconstructions = self.decoder(encoded_video, quant_b)

        return reconstructions

class LlamaConnector(nn.Module):
    def __init__(self, z_channels, out_channels) -> None:
        super().__init__()
        self.proj_in = CausalResnetBlockFactorized3d(
                    in_channels=z_channels,
                    out_channels=out_channels,
                    dropout=0,
                    num_groups=1,
                )
        config = LlamaConfig(
                hidden_size=out_channels,         
                intermediate_size=out_channels*2,   
                num_hidden_layers=12,    
                num_attention_heads=8,    
                max_position_embeddings=1024, 
                rms_norm_eps=1e-6,
                use_cache=True,
            )
        self.llama_model = LlamaModel(config=config)
    def forward(self, x):
        x = self.proj_in(x)
        x = x.flatten(2).permute(0, 2, 1)
        outputs = self.llama_model(inputs_embeds=x)
        x = outputs.last_hidden_state
        return x
    

class CausalDiscreteVideoLatentDynamicTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim
       
        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)
        # self.encoder = LDMEncoderFactorized(
        #     z_channels=z_factor * z_channels, act_embedding_num=act_embedding_num, **kwargs
        # )
        
        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)
        # self.decoder = LDMDecoderFactorized(
        #     z_channels=z_channels, **kwargs
        # )


        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)
        # import pdb;pdb.set_trace()
        # self.quant_to_dit_dim = CausalConv3d(z_channels, 1024, kernel_size=1, padding=0)
        connector_type = kwargs.get('connector_type', 'conv')
        self.connector_type = connector_type
        if connector_type == 'llama':
            self.quant_to_dit_dim = LlamaConnector(z_channels, out_channels=1024)
            total_params = sum(p.numel() for p in self.quant_to_dit_dim.parameters())
            logging.info(f"Connector Typ: Llama Transformers. Total parameters: {total_params / 1e9:.2f}B")
        else:
            self.quant_to_dit_dim = CausalResnetBlockFactorized3d(
                    in_channels=z_channels,
                    out_channels=1024,
                    dropout=0,
                    num_groups=1,
                )
            total_params = sum(p.numel() for p in self.quant_to_dit_dim.parameters())
            logging.info(f"Connector Typ: ConvBlock. Total parameters: {total_params / 1e9:.2f}B")

        quantizer_name = kwargs.get("quantizer", DiscreteQuantizer.RESFSQ.name)
        if quantizer_name == DiscreteQuantizer.VQ.name:
            assert "num_embeddings" in kwargs, f"`num_embeddings` must be provided for {quantizer_name}."
            kwargs.update(dict(embedding_dim=embedding_dim))
        elif quantizer_name == DiscreteQuantizer.LFQ.name:
            assert "codebook_size" in kwargs, f"`codebook_size` must be provided for {quantizer_name}."
            assert "codebook_dim" in kwargs, f"`codebook_dim` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.FSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.RESFSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
            assert "num_quantizers" in kwargs, f"`num_quantizers` must be provided for {quantizer_name}."
        self.quantizer = DiscreteQuantizer[quantizer_name].value(**kwargs)
        logging.info(f"{self.name} based on {quantizer_name}-VAE, with {kwargs}.")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")
        # self.loss_scale = 1.0
        # self.loss_reduce = 'mean'

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(CausalDiscreteVideoLatentDynamicTokenizer, self).to(torch.bfloat16)

    def encoder_jit(self):
        # return nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("encoder", self.encoder),
        #             ("quant_conv", self.quant_conv),
        #             ("quantizer", self.quantizer),
        #         ]
        #     )
        # )
        return CustomEncoderJIT(self)

    def decoder_jit(self):
        # return nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("inv_quant", InvQuantizerJit(self.quantizer)),
        #             ("post_quant_conv", self.post_quant_conv),
        #             ("decoder", self.decoder),
        #         ]
        #     )
        # )
        return CustomDecoderJIT(self)
    
    def last_decoder_layer(self):
        return self.decoder.conv_out

    def encode(self, x):
        # import pdb;pdb.set_trace()
        h, query = self.encoder(x)

        query = self.quant_conv(query) #B, 16, T-1, 1, 1
        quant_query = self.quantizer(query)
        quant_info, quant_codes, quant_loss = quant_query

        return h, quant_info, (quant_codes, quant_loss), query

    def decode(self, encoded_video, quant):
        quant = self.post_quant_conv(quant)
        
        return self.decoder(encoded_video, quant)

    def decode_code(self, encoded_video, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(encoded_video, quant_b)

    def forward(self, input, input_aug=None):
        # import pdb;pdb.set_trace()
        if input_aug is None:
            encoded_video, quant_info, (quant_codes, quant_loss), query = self.encode(input)
            decode_output = self.decode(encoded_video, quant_codes) #quant_codes: torch.Size([1, 6, 4, 1, 1]) 
            if isinstance(decode_output, tuple):
                reconstructions, mid_h = decode_output
            else:
                reconstructions = decode_output
                mid_h = None

            if self.training:
                return dict(
                    reconstructions=reconstructions,
                    quant_loss=quant_loss,
                    quant_info=quant_info,
                    quant_codes=quant_codes,
                    mid_h=mid_h,
                )
            return NetworkEval(
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
                quant_codes=quant_codes,
                ldm_query=query,
                mid_h=mid_h,
            )
        else:
            tensor_batch_aug_0 = input
            tensor_batch_aug_1 = input_aug
            encoded_video_aug_f0, _ = self.encoder(tensor_batch_aug_0)
            _, quant_info, (quant_codes, quant_loss), _ = self.encode(tensor_batch_aug_1)
            decode_output = self.decode(encoded_video_aug_f0, quant_codes) #quant_codes: torch.Size([1, 6, 4, 1, 1]) 
            if isinstance(decode_output, tuple):
                reconstructions, mid_h = decode_output
            else:
                reconstructions = decode_output
                mid_h = None

            output_dict =  dict(
                # reconstructions_aug=reconstructions_aug,
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
                quant_codes=quant_codes,
                mid_h=mid_h,
            )
            return output_dict


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .constants import TOKEN_LENGTH, DOWNLOAD_ROOT
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.cls_to_names import *

import copy

_tokenizer = _Tokenizer()

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding[:TOKEN_LENGTH]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion

        self.l2_norm_cal = False
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):

        image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        #### c-tpt
        if self.l2_norm_cal:
            prompt_mean = text_features.mean(0)
            feature_distance = text_features - prompt_mean
            l2_norm = torch.linalg.norm(feature_distance, dim=-1)
            l2_norm_mean = l2_norm.mean()
            
            #for saving to csv file
            self.l2_norm_mean = l2_norm_mean.item()
            
            #for training
            self.l2_norm_mean_training = l2_norm_mean
        
        ####
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)
        
    def forward_features(self, input):
        image_features = self.image_encoder(input.type(self.dtype))
        text_features = self.get_text_features()       
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return image_features, text_features, logit_scale
    
    def reset_image_encoder(self):
        self.image_encoder.load_state_dict(self.image_encoder_state)

    def reset_text_encoder(self):
        self.text_encoder.load_state_dict(self.text_encoder_state)


def get_coop(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

###################

class Shifter(nn.Module):
    def __init__(self, shift_init):
        super().__init__()
        self.register_buffer('shift_init_state', shift_init.clone().detach())
        self.shift = nn.Parameter(shift_init.clone())

    def reset(self):
        with torch.no_grad():
            self.shift.copy_(self.shift_init_state)

    def forward(self, input):
        return input+self.shift

class ClipShifterTestTimeTuning_v2(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14", n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, ):
        super(ClipShifterTestTimeTuning_v2, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.classnames = classnames

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    def add_shifter(self):
        with torch.no_grad():
            shift_init = torch.zeros([len(self.classnames), self.image_encoder.output_dim], dtype=self.dtype).to(self.device)
            self.text_embedding_init = self.get_text_features()
        self.shifter = Shifter(shift_init)

    def reset(self):
        self.shifter.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)


    def inference(self, image):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.shifter(self.text_embedding_init)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


    def forward(self, input):
        return self.inference(input)
        
    def forward_features(self, input):
        with torch.no_grad():
            image_features = self.image_encoder(input.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.shifter(self.text_embedding_init)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
        return image_features, text_features, logit_scale
    

def get_shift_v2(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):

    model = ClipShifterTestTimeTuning_v2(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

###################

class LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r, lora_alpha, original_weight, lora_dropout=0, original_bias=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.original_weight = nn.Parameter(original_weight.clone(), requires_grad=False)
        if original_bias is not None:
            self.original_bias = nn.Parameter(original_bias.clone(), requires_grad=False)
        else:
            self.register_parameter('original_bias', None)

        self.A = nn.Parameter(torch.empty(d_in, r))
        self.B = nn.Parameter(torch.zeros(r, d_out))
        nn.init.xavier_normal_(self.A)
        self.A_init_state = self.A.data.detach().clone()

    def forward(self, x):
        orig_output = F.linear(x, self.original_weight, self.original_bias)
        lora_output = F.linear(self.lora_dropout(F.linear(x, self.A.T)), self.B.T) * self.scaling
        return orig_output + lora_output

    def reset(self):
        self.A.data.copy_(self.A_init_state)
        self.B.data.zero_()

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, original_attn, r, lora_alpha, original_weight, lora_dropout=0):
        super().__init__()
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.r = r

        in_proj_weight = original_attn.in_proj_weight
        in_proj_bias = original_attn.in_proj_bias

        q_weight = in_proj_weight[:self.embed_dim, :]
        k_weight = in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        v_weight = in_proj_weight[2*self.embed_dim:, :]

        q_bias = in_proj_bias[:self.embed_dim] if in_proj_bias is not None else None
        k_bias = in_proj_bias[self.embed_dim:2*self.embed_dim] if in_proj_bias is not None else None
        v_bias = in_proj_bias[2*self.embed_dim:] if in_proj_bias is not None else None

        self.q_proj = LoRALinear(self.embed_dim, self.embed_dim, r, lora_alpha, q_weight, lora_dropout, q_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj.weight = nn.Parameter(k_weight, requires_grad=False)
        self.k_proj.bias = nn.Parameter(k_bias, requires_grad=False) if k_bias is not None else None
        self.v_proj = LoRALinear(self.embed_dim, self.embed_dim, r, lora_alpha, v_weight, lora_dropout, v_bias)

        self.out_proj = original_attn.out_proj
        self.out_proj.weight.requires_grad_(False)
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad_(False)

    def forward(self, x_q, x_k, x_v, need_weights=None, attn_mask=None):
        seq_len, batch_size, _ = x_q.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        # q = q.reshape(x_q.shape[0], x_q.shape[1], self.num_heads, self.head_dim)
        # import ipdb; ipdb.set_trace()
        
        # q = q.reshape(x_q.shape[0], x_q.shape[1], self.num_heads, self.head_dim).permute(2, 0, 1, 3).squeeze()
        # k = k.reshape(x_k.shape[0], x_k.shape[1], self.num_heads, self.head_dim).permute(2, 0, 1, 3).squeeze()
        # v = v.reshape(x_v.shape[0], x_v.shape[1], self.num_heads, self.head_dim).permute(2, 0, 1, 3).squeeze()

        q = q.reshape(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)  # (batch_size * num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, self.embed_dim)  # (seq_len, batch_size, embed_dim)
        output = self.out_proj(attn_output)  # (seq_len, batch_size, embed_dim)
        
        return output, None

class ClipLoraTestTimeTuning_v2(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14", n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, ):
        super(ClipLoraTestTimeTuning_v2, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def add_lora(self, layer_indices=[9,10,11], r=16, lora_alpha=32, lora_dropout=0.05):
        with torch.no_grad():
            self.text_embedding_init = self.get_text_features()
        for i in layer_indices:
            block = self.image_encoder.transformer.resblocks[i]
            original_attn = block.attn
            block.attn = LoRAMultiheadAttention(original_attn, r=r, lora_alpha=lora_alpha, original_weight=None, lora_dropout=lora_dropout)

    def reset(self):
        for module in self.image_encoder.modules():
            if isinstance(module, LoRALinear):
                module.reset()

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)
        
    def get_lora_params(self):
        params = []
        for module in self.image_encoder.modules():
            if isinstance(module, LoRALinear):
                params.append({'params': module.A})
                params.append({'params': module.B})
        return params
    
    def forward(self, input):
        image_features = self.image_encoder(input.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_embedding_init

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    def forward_features(self, input):
        with torch.no_grad():
            image_features = self.image_encoder(input.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.text_embedding_init

            logit_scale = self.logit_scale.exp()
        return image_features, text_features, logit_scale

def get_clip_lora_v2(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
    model = ClipLoraTestTimeTuning_v2(device, classnames, None, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)
    return model

class TextEncoder_Dynamic(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding[:TOKEN_LENGTH]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # x = prompts + self.positional_embedding.type(self.dtype)
        # pdb.set_trace()
        if len(prompts.size()) > 3:
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(prompts.size(0), 1, 1)
            tokenized_prompts = tokenized_prompts.view(-1, tokenized_prompts.size()[-1])
        x = prompts.view(-1, prompts.size()[-2], prompts.size()[-1]) + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_Dynamic(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=4, ctx_init=None, ctx_position='end', learned_cls=False, num_p=1
                 ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.num_p = num_p
        
        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx0 = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            ctx_vectors = torch.zeros(embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].size(), dtype=dtype)
            self.init_ctx = embedding[0, 1 + (n_ctx0-n_ctx) : 1 + n_ctx0, :].clone()
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)

        ##### multiple prompts #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(num_p, 1, 1)  # (N, L, D)
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (L, N, D)
        ##### feature initialization #####
        if num_p > 1:
            ctx_vectors = ctx_vectors.permute(1, 0, 2)  # (N, L, D)

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.ctx_order = list(range(num_p))
        self.ctx_use = [0] * num_p

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx0 - n_ctx, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx0 :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_ctx0 = n_ctx0
        self.classnames = classnames

    def reset(self):
        # pdb.set_trace()
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        assert False

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx + self.init_ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        ################ mixture of prompts ################
        elif self.num_p > 1:
            prefix = prefix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)
            suffix = suffix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning_Dynamic(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, num_p=1):
        super(ClipTestTimeTuning_Dynamic, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder_Dynamic(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.num_p = num_p
        self.prompt_learner = PromptLearner_Dynamic(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, num_p)
        self.criterion = criterion
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def get_text_features(self, p_s=None):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        if p_s is not None:
            t_features = self.text_encoder(prompts[p_s-1], tokenized_prompts)
        else:
            t_features = self.text_encoder(prompts, tokenized_prompts)

        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image, p_s=None):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        self.image_features = image_features.mean(0)

        if self.num_p > 1 and p_s is not None:
            text_features = self.get_text_features(p_s)
        else:
            text_features = self.get_text_features()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input, p_s=None):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, p_s)

def get_coop_dynamic(clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False, num_p=1):

    model = ClipTestTimeTuning_Dynamic(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_p=num_p)

    return model


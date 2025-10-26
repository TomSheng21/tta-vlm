import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

import clip
# import siglip
# import clip64

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

class CLIPRewards(torch.nn.Module):
    def __init__(self, device, arch="ViT-B/16", clipscore_weight=2.5,
                    amplify_rewards=False, sample_k=5, reward_process=True, process_batch=False,
                    default_resolutions=224) -> None:
        """
        calculating CLIP Reward
        Args:
            clipscore_weight: weight for calculating CLIPScore
            reward_process: If ture, post-process the rewards, e.g., subtract the reward mean or standardization
            amplify_rewards: If true, after subtracting the reward mean, also divide rewards by standard variance of rewards, i.e, standardization.
            sample_k: K for sampling.
            process_batch: If true, post-process the rewards within the {BatchSize x K} sampled text-image pairs.
                Others, post-process the rewards within the {1 x K} sampled text-image pairs.
                TPT augment the images, so we have a batch of augmented images from a single image.
        """
        super().__init__()
        self.default_resolutions = default_resolutions
        if arch == 'ViT-B/16-SigLip':
            self.clip_model, self.embed_dim, self.preprocess = clip64.load('ViT-B/16', device=device, download_root=clip64.DOWNLOAD_ROOT)
            self.clip_model.positional_embedding.data = self.clip_model.positional_embedding.data[:clip64.TOKEN_LENGTH]
        else:
            self.clip_model, self.embed_dim, self.preprocess = clip.load(arch, device=device, download_root=clip.DOWNLOAD_ROOT)
            self.clip_model.positional_embedding.data = self.clip_model.positional_embedding.data[:clip.TOKEN_LENGTH]
        self.resolutions = self.clip_model.visual.input_resolution
        self.clipscore_weight = clipscore_weight
        self.device = device
        self.class_features = None
        self.image_features = None
        self.amplify_rewards = amplify_rewards
        self.sample_k = sample_k
        self.reward_process = reward_process
        self.process_batch = process_batch
        self.clip_model.eval()

        print("\n CLIPRewards model created: \n"
                "\t visual backbone: {}, resolutions: {}, amplify_rewards: {}, sample_k: {}, \n"
                "\t reward_process: {}, process_batch: {}\n".format(
                    arch, self.resolutions, amplify_rewards, sample_k, reward_process, process_batch))

    @torch.no_grad()
    def set_class_features(self, classnames=None, tokenized_classes=None):
        self.class_features = self.extract_text_features(captions=classnames, tokenized_cap=tokenized_classes)

    @torch.no_grad()
    def set_image_features(self, images):
        self.image_features = self.extract_image_features(images)

    @torch.no_grad()
    def confidence_gap(self, predictions):
        """
        Args:
            predictions: shape [bs, C]
        """
        value, index = torch.topk(predictions, 2, dim=-1)
        gap = value[:, 0] - value[:, 1]
        gap = gap - torch.mean(gap)

        return gap

    @torch.no_grad()
    def CLIPScore(self, class_index, images=None, image_features=None, captions=None, tokenized_cap=None, text_features=None,
                        pairwise=True):
        """
        class_index: sampled class index
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        text_features = self.class_features[class_index]
        image_features = torch.repeat_interleave(self.image_features, self.sample_k, dim=0)

        if pairwise:
            similarity = self.clipscore_weight * text_features @ image_features.t()
        else:
            similarity = self.clipscore_weight * torch.sum(text_features * image_features, dim=-1)

        scores = torch.maximum(similarity, torch.zeros_like(similarity)).squeeze()

        return scores

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features without normalization"""
        if self.resolutions != self.default_resolutions:
            images = nn.functional.interpolate(images, size=self.resolutions, mode='bicubic', align_corners=True)
        image_features = self.clip_model.encode_image(images).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        if captions is not None:
            caption_tokens = clip.tokenize(captions, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(caption_tokens).float()
        if tokenized_cap is not None:
            text_features = self.clip_model.encode_text(tokenized_cap).float()

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    @torch.no_grad()
    def rewards_post_process(self, clip_score):
        """
        clip_score: shape [bs, K] or [bs * K]
        """
        if clip_score.shape[-1] > 1 and self.reward_process:
            mean = torch.mean(clip_score, dim=-1, keepdim=True)
            if self.amplify_rewards:
                std = torch.std(clip_score, dim=-1, keepdim=True) + 1e-5
            else:
                std = 1.0
            clip_score = (clip_score - mean) / std

        return clip_score.flatten()

    @torch.no_grad()
    def calulate_similarity(self):
        """
        pairwise: if True, calculate the similarity between every image and text pairs
        """
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.class_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class RLCF():
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.sample_k = 3
        self.amplify_rewards = 0
        self.reward_process = 1
        self.process_batch = 0

        

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, lr=7e-3, weight_decay=5e-4)
    
        self.optim_state = deepcopy(self.optimizer.state_dict())

        ### reward model adopts the same architecture as the main model

        self.reward_model = CLIPRewards(self.device, arch=args.arch,
                                amplify_rewards=self.amplify_rewards, sample_k=self.sample_k,
                                reward_process=self.reward_process, process_batch=self.process_batch)

        self.reward_model.set_class_features(tokenized_classes=self.model.prompt_learner.tokenized_prompts)

        args.tta_steps = 3
        args.selection_p = 0.1
        args.min_entropy_reg = False

    def pre_adaptation(self):
        with torch.no_grad():
            self.model.reset()
        self.optimizer.load_state_dict(self.optim_state)
    
    def adaptation_process(self, image, images, args):

        assert args.tta_steps > 0

        selected_idx = None
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                if selected_idx is not None:
                    output = self.model(images[selected_idx])
                else:
                    output = self.model(images)
                    output, selected_idx = select_confident_samples(output, args.selection_p)
                    self.reward_model.set_image_features(images[selected_idx])
                bs = output.shape[0]

                # top-k sample results
                value, index = torch.topk(output, self.sample_k, dim=-1)
                flatten_index = index.flatten()
                # reward calculation
                clip_score = self.reward_model.CLIPScore(class_index=flatten_index, pairwise=False)     
                rewards = self.reward_model.rewards_post_process(clip_score if self.reward_model.process_batch else clip_score.reshape(bs, -1))


                rep_output = torch.repeat_interleave(output, self.sample_k, dim=0)
                all_loss = F.cross_entropy(rep_output, flatten_index, reduction='none')
                loss = torch.mean(rewards * all_loss)

                if args.min_entropy_reg:
                    loss = loss + args.min_entropy_w * avg_entropy(output)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tta_output = self.model(image)
                tta_features, _, _ = self.model.forward_features(images)
                tta_outputs = self.model(images)
                tta_prompt_embedding = self.model.prompt_learner.ctx.detach()

        return_dict = {
            'output':tta_output,
            'tta_features':tta_features,
            'tta_outputs':tta_outputs,
            'tta_prompt_embedding':tta_prompt_embedding,
        }

        return return_dict




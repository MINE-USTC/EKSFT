# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class EKSFTSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer for EKSFT (Entropy-KL Selective Fine-Tuning) training.
    
    EKSFT selectively fine-tunes tokens based on KL divergence and entropy metrics,
    enabling more efficient and targeted model adaptation.
    """

    def __init__(
        self,
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Initialize reference model for EKSFT
        self.ref_model = ref_model
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = DPOTrainer._prepare_deepspeed(self, self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def calculate_kl_and_entropy(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Calculate the KL divergence and entropy for each token position.
        
        Args:
            model: Current model being trained.
            ref_model: Reference model for KL divergence calculation.
            inputs: Input dictionary containing input_ids, attention_mask, etc.
            attention_mask: Mask for valid token positions.
            
        Returns:
            kl_per_token: KL divergence for each token (batch_size, seq_len).
            entropy_per_token: Entropy for each token (batch_size, seq_len).
            log_probs_model: Log probabilities from current model.
            log_probs_ref: Log probabilities from reference model.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs['input_ids'])
        
        # Disable gradient computation
        with torch.no_grad():
            logits_model = model(**inputs).logits
            logits_ref = ref_model(**inputs).logits
        
        # Calculate log probabilities (numerically stable)
        log_probs_model = F.log_softmax(logits_model, dim=-1)
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        
        # Calculate model probability distribution
        probs_model = log_probs_model.exp()
        
        # Calculate entropy: entropy = -Σ(p * log(p))
        entropy_per_token = -torch.sum(probs_model * log_probs_model, dim=-1)
        
        # Calculate KL divergence: kl = Σ(p_model * (log(p_model) - log(p_ref)))
        # Use clamp to avoid numerical issues from log(0)
        log_diff = (log_probs_model - log_probs_ref).clamp(min=-1e10, max=1e10)
        kl_per_token = torch.sum(probs_model * log_diff, dim=-1)
        
        # Apply attention mask (padding positions are zeroed)
        kl_per_token = kl_per_token * attention_mask
        entropy_per_token = entropy_per_token * attention_mask

        return kl_per_token, entropy_per_token, log_probs_model, log_probs_ref

    def create_intersection_mask(
        self,
        kl_per_token: torch.Tensor,
        entropy_per_token: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
        largest_kl: bool,
        largest_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Create intersection/union mask based on KL divergence and entropy.
        
        Args:
            kl_per_token: KL divergence tensor (batch_size, seq_len).
            entropy_per_token: Entropy tensor (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            top_k: Number of top positions to select.
            largest_kl: If True, select tokens with largest KL; otherwise smallest.
            largest_entropy: If True, select tokens with largest entropy; otherwise smallest.
            
        Returns:
            intersection_mask: Combined mask based on is_union_mask setting.
            top_kl_mask: Mask for top-k KL positions.
            top_entropy_mask: Mask for top-k entropy positions.
        """
        # Ensure input shapes are consistent
        assert kl_per_token.shape == entropy_per_token.shape == attention_mask.shape
        
        # If top_k is 0, return attention_mask
        if top_k == 0:
            return attention_mask, attention_mask, attention_mask
        
        # Clone tensors to avoid modifying original data
        kl = kl_per_token.clone()
        entropy = entropy_per_token.clone()
        
        # Prepare masks: set padding positions to extreme values
        if largest_kl:
            kl[attention_mask == 0] = float('-inf')
        else:
            kl[attention_mask == 0] = float('inf')
        if largest_entropy:
            entropy[attention_mask == 0] = float('-inf')
        else:
            entropy[attention_mask == 0] = float('inf')
        
        # Get total number of elements
        flat_size = kl.numel()
        
        # Ensure top_k does not exceed valid element count
        actual_top_k = min(top_k, flat_size)
        
        # Get top-k positions for KL
        top_kl_values, top_kl_indices = torch.topk(kl.view(-1), k=actual_top_k, largest=largest_kl)
        
        # Get top-k positions for entropy
        top_entropy_values, top_entropy_indices = torch.topk(entropy.view(-1), k=actual_top_k, largest=largest_entropy)
        
        # Safe index assignment function
        def safe_index_assign(tensor: torch.Tensor, indices: torch.Tensor, value: bool) -> torch.Tensor:
            valid_indices = indices[(indices >= 0) & (indices < tensor.numel())]
            tensor[valid_indices] = value
            return tensor
        
        # Create KL mask
        top_kl_mask = torch.zeros(flat_size, dtype=torch.bool, device=kl.device)
        top_kl_mask = safe_index_assign(top_kl_mask, top_kl_indices, True)
        
        # Create entropy mask
        top_entropy_mask = torch.zeros(flat_size, dtype=torch.bool, device=kl.device)
        top_entropy_mask = safe_index_assign(top_entropy_mask, top_entropy_indices, True)
        
        # Combine masks (union or intersection)
        if self.finetuning_args.eksft_is_union_mask:
            intersection_mask = top_kl_mask | top_entropy_mask
        else:
            intersection_mask = top_kl_mask & top_entropy_mask
        
        # Reshape to original shape
        intersection_mask = intersection_mask.view_as(kl_per_token)
        
        return intersection_mask, top_kl_mask, top_entropy_mask

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        r"""
        Compute loss with EKSFT (Entropy-KL Selective Fine-Tuning) strategy.
        
        The loss formula: total_loss = CE_loss - lambda_entropy * entropy_loss + lambda_kl * KL_loss
        
        - CE_loss is computed on masked positions (intersection/union of high KL and high/low entropy)
        - entropy_loss encourages exploration on entropy-selected positions
        - KL_loss regularizes deviation from reference model on KL-selected positions
        """
        finetuning_args = self.finetuning_args
        
        # Get EKSFT hyperparameters
        largest_kl = finetuning_args.eksft_largest_kl
        largest_entropy = finetuning_args.eksft_largest_entropy
        
        if self.ref_model is not None:
            # Get labels and create attention mask
            labels = inputs.get('labels', None)
            attention_mask = labels.ne(IGNORE_INDEX).to(labels.device)
            
            # Calculate KL divergence and entropy
            kl_per_token, entropy_per_token, log_probs_model, log_probs_ref = self.calculate_kl_and_entropy(
                model, self.ref_model, inputs, attention_mask
            )
            
            # Calculate batch KL and entropy sums for logging
            kl_sum = (kl_per_token * attention_mask).sum()
            entropy_sum = (entropy_per_token * attention_mask).sum()
            total_tokens = attention_mask.sum()
            kl_sum = self.accelerator.gather(kl_sum)
            entropy_sum = self.accelerator.gather(entropy_sum)
            total_tokens = self.accelerator.gather(total_tokens)
            
            # Log KL and entropy metrics
            if self.is_world_process_zero() and finetuning_args.eksft_output_dir is not None:
                global_kl_sum = kl_sum.cpu().float().numpy()
                global_entropy_sum = entropy_sum.cpu().float().numpy()
                global_total_tokens = total_tokens.cpu().float().numpy()

                global_kl_per_token = np.zeros_like(global_kl_sum)
                global_entropy_per_token = np.zeros_like(global_entropy_sum)
                mask = global_total_tokens > 0
                global_kl_per_token[mask] = global_kl_sum[mask] / global_total_tokens[mask]
                global_entropy_per_token[mask] = global_entropy_sum[mask] / global_total_tokens[mask]
                
                os.makedirs(finetuning_args.eksft_output_dir, exist_ok=True)
                path = os.path.join(finetuning_args.eksft_output_dir, "kl_entropy.jsonl")
                with open(path, "a") as log_file:
                    log_data = {
                        "step": self.state.global_step,
                        "kl": float(global_kl_sum.mean().item()),
                        "entropy": float(global_entropy_sum.mean().item()),
                        "kl_per_token": float(global_kl_per_token.mean().item()),
                        "entropy_per_token": float(global_entropy_per_token.mean().item())
                    }
                    log_file.write(json.dumps(log_data) + "\n")
            
            # Calculate top-k value
            top_k_ratio = finetuning_args.eksft_top_k_ratio
            total_valid_tokens = attention_mask.sum().item()
            k = max(0, int(total_valid_tokens * top_k_ratio))
            
            if k > 0:
                mask_attention, kl_mask, entropy_mask = self.create_intersection_mask(
                    kl_per_token, entropy_per_token, attention_mask, k, largest_kl, largest_entropy
                )
            else:
                mask_attention = attention_mask
                kl_mask = torch.zeros_like(attention_mask, dtype=torch.bool).view(-1)
                entropy_mask = torch.zeros_like(attention_mask, dtype=torch.bool).view(-1)
            
            # Clear memory
            torch.cuda.empty_cache()
        else:
            # Fallback to standard SFT if no reference model
            return super().compute_loss(model, inputs, *args, **kwargs)
        
        # Get model outputs
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        probs = torch.softmax(logits, dim=-1)
        del outputs
        torch.cuda.empty_cache()
        
        labels = inputs.labels.clone()
        
        # Apply mask for CE loss
        if self.ref_model is not None and k > 0:
            labels_ce = labels.masked_fill(mask_attention, IGNORE_INDEX)
        else:
            labels_ce = labels.clone()
        
        ce_loss = model.loss_function(logits, labels_ce, vocab_size=vocab_size)
        
        lambda_entropy = finetuning_args.eksft_lambda_entropy
        lambda_kl = finetuning_args.eksft_lambda_kl
        
        # ================== Entropy regularization loss (on entropy positions) ==================
        if entropy_mask.any() and lambda_entropy > 0:
            max_entropy = torch.log(torch.tensor(vocab_size, device=logits.device))
            token_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1) / max_entropy
            entropy_mask_2d = entropy_mask.view_as(token_entropy) if entropy_mask.dim() == 1 else entropy_mask
            entropy_loss = (token_entropy * entropy_mask_2d.float()).sum() / (entropy_mask.sum() + 1e-12)
        else:
            entropy_loss = torch.tensor(0.0, device=logits.device)

        # ================== KL regularization loss (on KL positions) ==================
        if kl_mask.any() and lambda_kl > 0:
            log_diff = (probs - log_probs_ref).clamp(min=-1e10, max=1e10)
            probs_model = probs
            kl_per_token_loss = torch.sum(probs_model * log_diff, dim=-1)
            kl_per_token_loss = kl_per_token_loss * attention_mask
            kl_mask_2d = kl_mask.view_as(kl_per_token_loss) if kl_mask.dim() == 1 else kl_mask
            kl_loss = (kl_per_token_loss * kl_mask_2d.float()).sum() / (kl_mask.sum() + 1e-12)
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)

        # ================== Dynamic lambda adjustment ==================
        # Re-read lambda values (matching original behavior)
        lambda_entropy = finetuning_args.eksft_lambda_entropy
        lambda_kl = finetuning_args.eksft_lambda_kl
        target_entropy = finetuning_args.eksft_target_entropy
        target_kl = finetuning_args.eksft_target_kl
        
        if entropy_loss.item() < target_entropy:
            lambda_entropy *= 1.05
        else:
            lambda_entropy *= 0.95
                
        if kl_loss.item() > target_kl:
            lambda_kl *= 1.05
        else:
            lambda_kl *= 0.95

        # ================== Combine total loss ==================
        # total_loss = CE - lambda_entropy * entropy + lambda_kl * KL
        total_loss = ce_loss - lambda_entropy * entropy_loss + lambda_kl * kl_loss
        
        # Log loss components
        jaccard_index = (kl_mask & entropy_mask).sum().item() / (kl_mask | entropy_mask).sum().item()
        kl_entropy_IoU = torch.tensor(jaccard_index).to(total_loss.device)
        
        kl_entropy_IoU = self.accelerator.gather(kl_entropy_IoU.detach())
        global_ce_loss = self.accelerator.gather(ce_loss.detach())
        global_entropy_loss = self.accelerator.gather(entropy_loss.detach())
        global_kl_loss = self.accelerator.gather(kl_loss.detach())
        
        if self.is_world_process_zero():
            global_ce_loss = global_ce_loss.cpu().float().numpy()
            global_entropy_loss = global_entropy_loss.cpu().float().numpy()
            global_kl_loss = global_kl_loss.cpu().float().numpy()
            kl_entropy_IoU = kl_entropy_IoU.cpu().float().numpy()

            if finetuning_args.eksft_output_dir is not None:
                path = os.path.join(finetuning_args.eksft_output_dir, "other_information_multiloss.jsonl")
                with open(path, "a") as log_file:
                    log_data = {
                        "step": self.state.global_step,
                        "ce_loss": float(global_ce_loss.mean().item()),
                        "entropy_loss": float(global_entropy_loss.mean().item()),
                        "kl_loss": float(global_kl_loss.mean().item()),
                        "kl_entropy_IoU": float(kl_entropy_IoU.mean().item())
                    }
                    log_file.write(json.dumps(log_data) + "\n")
        
        # Print loss components for debugging (matching original print statements)
        print("mask_ce_loss: " + str(ce_loss.item()))
        print("entropy_loss: " + str((lambda_entropy * entropy_loss).item()))
        print("kl_loss: " + str((lambda_kl * kl_loss).item()))

        return total_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

import gc
import os
import argparse
from dataclasses import asdict

import torch
from accelerate.state import AcceleratorState
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
    LlamaForCausalLM,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
)
from transformers.utils import logging
from arguments import (
    MHA2MLAModelArguments,
    MHA2MLADataArguments,
    MHA2MLATrainingArguments,
)
from progressive_args import ProgressiveCompressionArguments
from helpers import load_dataset, load_optimizer_scheduler, freeze_non_attn_weights
from patching_model_load import patch_model
from patching_qwen2 import mha2mla_qwen2
from patching_qwen3 import mha2mla_qwen3
from patching_qwen3_moe import mha2mla_qwen3_moe
from patching_llama import mha2mla_llama
from recompress import recompress_model

logger = logging.get_logger(__name__)


def main():
    # ── 1. Parse arguments (4 dataclasses) ──
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True)
    cfg = cfg_parser.parse_args()
    hf_parser = HfArgumentParser((
        MHA2MLATrainingArguments,
        MHA2MLAModelArguments,
        MHA2MLADataArguments,
        ProgressiveCompressionArguments,
    ))
    train_args, mha2mla_args, data_args, prog_args = hf_parser.parse_yaml_file(
        cfg.cfg_file
    )

    rank_schedule = prog_args.rank_schedule
    steps_per_stage = prog_args.steps_per_stage
    warmup_steps_per_stage = prog_args.warmup_steps_per_stage
    num_stages = len(rank_schedule)
    total_steps = sum(steps_per_stage)

    print(f"Progressive compression: {num_stages} stages")
    print(f"  rank_schedule:          {rank_schedule}")
    print(f"  steps_per_stage:        {steps_per_stage}")
    print(f"  warmup_steps_per_stage: {warmup_steps_per_stage}")
    print(f"  reset_optimizer:        {prog_args.reset_optimizer}")
    print(f"  total_steps:            {total_steps}")

    # ── 2. Load tokenizer and model ──
    name = mha2mla_args.model_name_or_path
    model_args = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mha_model = AutoModelForCausalLM.from_pretrained(name)
    print(mha_model)

    # ── 3. Initial patch with rank_schedule[0] ──
    mha2mla_args.low_rank = rank_schedule[0]
    mla_model, q_idx, k_idx = patch_model(mha_model, model_args, mha2mla_args)

    # ── 4. Apply forward patches (once) ──
    if isinstance(mha_model, LlamaForCausalLM):
        mha2mla_llama(q_idx, k_idx)
    elif isinstance(mha_model, Qwen2ForCausalLM):
        mha2mla_qwen2(q_idx, k_idx)
    elif isinstance(mha_model, Qwen3ForCausalLM):
        mha2mla_qwen3(q_idx, k_idx)
    elif isinstance(mha_model, Qwen3MoeForCausalLM):
        mha2mla_qwen3_moe(q_idx, k_idx)

    model = mla_model
    if train_args.is_freeze_non_attn:
        freeze_non_attn_weights(model)
    model.config.mha2mla = asdict(mha2mla_args)

    if train_args.bf16:
        model = model.to(dtype=torch.bfloat16)
    elif train_args.fp16:
        model = model.to(dtype=torch.float16)

    # ── 5. Load dataset once (with total_steps for proper sizing) ──
    original_max_steps = train_args.max_steps
    train_args.max_steps = total_steps
    train_dataset = load_dataset(data_args, train_args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    base_output_dir = train_args.output_dir

    # ── 6. Stage loop ──
    for stage_idx in range(num_stages):
        current_rank = rank_schedule[stage_idx]
        current_steps = steps_per_stage[stage_idx]
        current_warmup = warmup_steps_per_stage[stage_idx]

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}/{num_stages - 1}: "
              f"rank={current_rank}, steps={current_steps}, warmup={current_warmup}")
        print(f"{'='*60}")

        # 6a. Clean up previous Trainer's accelerator state & recompress
        if stage_idx > 0:
            # Must reset accelerator singleton so a new Trainer can
            # re-initialize DeepSpeed for this stage.
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            AcceleratorState._reset_state()

            recompress_model(
                model=model,
                new_low_rank=current_rank,
                num_kv_heads=model_args.num_key_value_heads,
                svd_method=mha2mla_args.svd_init_method,
            )
            # Update dtype after recompression (new params are float32)
            if train_args.bf16:
                model = model.to(dtype=torch.bfloat16)
            elif train_args.fp16:
                model = model.to(dtype=torch.float16)

        # 6b. Configure training args for this stage
        train_args.max_steps = current_steps
        train_args.output_dir = os.path.join(base_output_dir, f"stage_{stage_idx}")
        train_args.run_name = f"{train_args.run_name}-stage{stage_idx}" if stage_idx == 0 else train_args.run_name.rsplit("-stage", 1)[0] + f"-stage{stage_idx}"

        # Configure lr schedule for this stage: warmup -> constant
        train_args.lr_scheduler_kwargs = {
            "lr_decay_starting_step": current_steps,
            "lr_decay_steps": 0,
            "lr_decay_style": "1-sqrt",
            "lr_warmup_steps": current_warmup,
            "lr_warmup_style": "linear",
            "min_decay_lr": 0,
        }

        # 6c. Create new optimizer and scheduler
        optimizer, lr_scheduler = load_optimizer_scheduler(model, train_args)

        # 6d. Create Trainer and train
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, lr_scheduler),
            data_collator=data_collator,
        )
        trainer.train()

        # Log stage info
        stage_info = {
            **asdict(mha2mla_args),
            "progressive_stage": stage_idx,
            "progressive_rank": current_rank,
        }
        trainer.log(stage_info)

        # 6e. Save stage checkpoint
        print(f"Stage {stage_idx} complete. Saving checkpoint to {train_args.output_dir}")
        trainer.save_model(train_args.output_dir)

    # ── 7. Save final model ──
    final_output_dir = os.path.join(base_output_dir, "final")
    print(f"\nSaving final model to {final_output_dir}")
    state_dict = model.state_dict()
    if model.config.tie_word_embeddings:
        state_dict["lm_head.weight"] = state_dict[
            "model.embed_tokens.weight"
        ].clone()
    model.save_pretrained(final_output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()

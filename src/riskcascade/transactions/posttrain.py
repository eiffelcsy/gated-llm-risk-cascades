"""GRPO post-training of the dossier-mode LLM investigator with a cost-based reward.

Prompts are the tier-2 training block (train period only, near the threshold),
in exactly the format InvestigationAgent(mode="dossier") uses at inference.

Reward, with abstaining as the safe zero:
  correct block of fraud   +c_fn     wrong allow of fraud    -c_fn
  correct allow of honest  +c_fp     wrong block of honest   -c_fp
  abstain                   0        unparseable answer      -c_fp
Answering beats abstaining only when the expected reward is positive, i.e.
block when P(fraud) > c_fp/(c_fn+c_fp) and allow when P(honest) > c_fn/(c_fn+c_fp):
the same decision rule the override gate applies.

Usage (needs a GPU for real runs; LoRA keeps memory small):
    python -m riskcascade.transactions.posttrain --config configs/transactions.yaml
Then serve output_dir with vLLM (--enable-lora) and set tier2.arm=llm_dossier, tier2.llm=<adapter name>.
"""
from __future__ import annotations

import argparse

import yaml

from ..core.llm import parse_json
from .agent import dossier, dossier_messages, dossier_text


def cost_reward(text: str, label: int, c_fn: float, c_fp: float) -> float:
    out = parse_json(text)
    v = out.get("verdict")
    if v not in (0, 1, -1):
        return -c_fp
    if v == -1:
        return 0.0
    stake = c_fn if label == 1 else c_fp  # what is gained or lost on this case
    return stake if v == label else -stake


def make_reward_fn(c_fn: float, c_fp: float):
    """TRL reward signature: dataset columns arrive as keyword lists."""
    def reward(completions, label, **_):
        return [cost_reward(c[0]["content"] if isinstance(c, list) else c, int(y), c_fn, c_fp)
                for c, y in zip(completions, label)]
    return reward


def build_dataset(p, cfg: dict):
    from datasets import Dataset

    rows = p.df.iloc[p.tier2_idx]
    scores = p.scorer.score(rows)
    prompts = [dossier_messages(dossier_text(rows.iloc[[i]], s, dossier(p.tools, rows.iloc[[i]])))
               for i, s in enumerate(scores)]
    return Dataset.from_dict({"prompt": prompts, "label": p.y[p.tier2_idx].tolist()})


def train(cfg: dict) -> None:
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from .pipeline import prepare

    pt, c = cfg["posttrain"], cfg["costs"]
    args = GRPOConfig(output_dir=pt["output_dir"], learning_rate=pt["learning_rate"], max_steps=pt["max_steps"],
                      num_generations=pt["num_generations"], per_device_train_batch_size=pt["num_generations"],
                      max_completion_length=pt["max_completion_length"], logging_steps=10, report_to="none")
    trainer = GRPOTrainer(model=pt["base_model"], reward_funcs=make_reward_fn(c["c_fn"], c["c_fp"]), args=args,
                          train_dataset=build_dataset(prepare(cfg), cfg),
                          peft_config=LoraConfig(r=pt["lora_r"], lora_alpha=2 * pt["lora_r"],
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(pt["output_dir"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transactions.yaml")
    train(yaml.safe_load(open(ap.parse_args().config)))

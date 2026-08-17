#!/usr/bin/env python3
"""
Filter SFT parquet by token length.

Usage:
    python mine/filter_sft_data.py

Filters:
    - user content   <= 512 tokens
    - assistant content <= 512 tokens
"""

import argparse
import os

import datasets
from transformers import AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT = "data/sft_train.parquet"
DEFAULT_OUTPUT = "data/sft_train_filtered.parquet"


def token_length_filter(tokenizer, max_user: int, max_asst: int):
    """Return a filter function for HuggingFace Dataset.filter()."""
    def _filter(example):
        msgs = example["messages"]
        user_tokens = tokenizer.tokenize(msgs[0]["content"])
        asst_tokens = tokenizer.tokenize(msgs[1]["content"])
        return len(user_tokens) <= max_user and len(asst_tokens) <= max_asst
    return _filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SFT dataset by token length")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Input parquet path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output parquet path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Tokenizer model (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--max_user_tokens", type=int, default=512,
                        help="Max tokens for user content (default: 512)")
    parser.add_argument("--max_asst_tokens", type=int, default=512,
                        help="Max tokens for assistant content (default: 512)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.input}")
    ds = datasets.Dataset.from_parquet(args.input)
    total = len(ds)
    print(f"  Total samples: {total}")

    print(f"Filtering: user <= {args.max_user_tokens} tokens, "
          f"assistant <= {args.max_asst_tokens} tokens")
    filtered = ds.filter(
        token_length_filter(tokenizer, args.max_user_tokens, args.max_asst_tokens)
    )
    kept = len(filtered)
    removed = total - kept
    print(f"  Kept:    {kept} ({100*kept/total:.1f}%)")
    print(f"  Removed: {removed} ({100*removed/total:.1f}%)")

    filtered.to_parquet(args.output)
    print(f"\n✅ Filtered dataset saved to: {args.output}")

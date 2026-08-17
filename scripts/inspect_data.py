#!/usr/bin/env python3
"""
Inspect SFT / RL parquet datasets — print statistics, then interactively view samples.

Usage:
    python scripts/inspect_data.py data/sft_train_merged.parquet
    python scripts/inspect_data.py data/rl_train.parquet
"""

import argparse
import sys
from collections import Counter

import datasets


def print_stats(ds: datasets.Dataset, fmt: str) -> None:
    """Print summary statistics — always shown first."""
    print(f"\n{'='*60}")
    print(f"  📊 数据集统计")
    print(f"{'='*60}")
    print(f"  格式:     {fmt}")
    print(f"  总条数:   {len(ds)}")

    if fmt == "SFT":
        extra = ds["extra_info"]
        types = Counter(e.get("type", "?") for e in extra)
        print(f"  类型分布:")
        for t, c in types.most_common():
            print(f"           {t}: {c}")

        # check for domain field
        if "domain" in (extra[0] or {}):
            domains = []
            for e in extra:
                d = e.get("domain")
                if isinstance(d, list):
                    domains.extend(d)
                elif d:
                    domains.append(d)
            if domains:
                dc = Counter(domains)
                print(f"  领域分布:")
                for d, c in dc.most_common():
                    print(f"           {d}: {c}")

        # data source breakdown
        sources = Counter(e.get("data_source", "?") for e in ds)
        if len(sources) > 1:
            print(f"  数据来源:")
            for s, c in sources.most_common():
                print(f"           {s}: {c}")

    elif fmt == "RL/Test":
        extra = ds["extra_info"]
        types = Counter(e.get("type", "?") for e in extra)
        splits = Counter(e.get("split", "?") for e in extra)
        print(f"  split 分布: {dict(splits)}")
        print(f"  类型分布:")
        for t, c in types.most_common():
            print(f"           {t}: {c}")

    # Show token length stats for SFT
    if fmt == "SFT":
        user_lens = [len(item["messages"][0]["content"]) for item in ds]
        asst_lens = [len(item["messages"][1]["content"]) for item in ds]
        print(f"  User 长度:      min={min(user_lens)}  max={max(user_lens)}  avg={sum(user_lens)//len(user_lens)}")
        print(f"  Assistant 长度: min={min(asst_lens)}  max={max(asst_lens)}  avg={sum(asst_lens)//len(asst_lens)}")


def print_sample_full(ds: datasets.Dataset, idx: int, fmt: str) -> None:
    """Print a single sample in full."""
    item = ds[idx]

    print(f"\n{'='*60}")
    print(f"  📝 Sample [{idx}]  —  完整内容")
    print(f"{'='*60}")

    if fmt == "SFT":
        msgs = item["messages"]
        info = item.get("extra_info", {})

        qid = info.get("question_id") or info.get("sample_id", "?")
        qtype = info.get("type", "?")
        gt = info.get("correct_answer", "?")

        print(f"  id:           {qid}")
        print(f"  type:         {qtype}")
        print(f"  data_source:  {item.get('data_source', '?')}")
        print(f"  correct_answer: {gt}")
        print(f"\n  ── User ──")
        print(msgs[0]["content"])
        print(f"\n  ── Assistant (CoT) ──")
        print(msgs[1]["content"])

    elif fmt == "RL/Test":
        prompt = item["prompt"][0]["content"]
        reward = item.get("reward_model", {})
        info = item.get("extra_info", {})

        qid = info.get("question_id") or info.get("sample_id", "?")
        qtype = info.get("type", "?")
        split = info.get("split", "?")
        gt = reward.get("ground_truth", "?")

        print(f"  split:        {split}")
        print(f"  id:           {qid}")
        print(f"  type:         {qtype}")
        print(f"  ground_truth: {gt}")
        print(f"\n  ── Prompt ──")
        print(prompt)

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect verl parquet datasets")
    parser.add_argument("path", type=str, help="Path to .parquet file")
    parser.add_argument("-i", type=int, default=None, help="Directly view sample at index (skip interactive mode).")
    args = parser.parse_args()

    ds = datasets.Dataset.from_parquet(args.path)

    sample = ds[0]
    if "messages" in sample:
        fmt = "SFT"
    elif "prompt" in sample:
        fmt = "RL/Test"
    else:
        print(f"Unknown format.  Keys: {list(sample.keys())}")
        sys.exit(1)

    # Always print stats first
    print_stats(ds, fmt)

    # Direct mode: -i N
    if args.i is not None:
        if 0 <= args.i < len(ds):
            print_sample_full(ds, args.i, fmt)
        else:
            print(f"\n❌ Index out of range (0 ~ {len(ds)-1})")
        sys.exit(0)

    # Interactive mode
    print(f"\n  输入样本索引 (0 ~ {len(ds)-1}) 查看完整内容，输入 q 退出")
    while True:
        try:
            raw = input("  >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw.lower() in ("q", "quit", "exit", ""):
            break

        try:
            idx = int(raw)
            if 0 <= idx < len(ds):
                print_sample_full(ds, idx, fmt)
                print(f"  输入下一个索引 (0 ~ {len(ds)-1})，或 q 退出")
            else:
                print(f"  ❌ 超出范围，请输入 0 ~ {len(ds)-1}")
        except ValueError:
            print(f"  ❌ 无效输入，请输入数字或 q 退出")

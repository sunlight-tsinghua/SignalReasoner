# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

import re

# 限制检查的字符数，只关注输出尾部（通常最终答案在末尾）
_SOLUTION_CLIP_CHARS = 500


def extract_boxed_answers(text: str) -> list:
    """
    从文本中提取所有 \\boxed{...} 表达式中的内容。
    """
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]


def extract_boxed_answers_ground_truth(ground_truth: str) -> list:
    """从标准答案字符串中提取所有 \\boxed{} 内容。"""
    return extract_boxed_answers(ground_truth)


def normalize_math_expression(s: str) -> str:
    """移除字符串中所有空白字符（包括内部空格）。"""
    return re.sub(r'\s+', '', s)


def compute_score(
    solution_str: str,
    ground_truth: str,
    correct_score: float = 1.0,
    partial_score: float = 0.2,
    format_score: float = 0.1,
    **kwargs,
) -> float:
    """
    Reward function for Wireless fill-in-the-blank GRPO training.

    - correct_score (1.0): boxed-answer list EXACTLY equals the ground truth
      (same length, position-wise equal after whitespace normalization).
    - partial_score (0.2): some (not all) ground-truth answers matched.
    - format_score (0.1): output contains a \\boxed{...} but nothing correct.
    - 0.0: no \\boxed{} at all.

    Full credit requires strict one-to-one matching, which avoids the reward-hack
    of membership matching ("every GT \\boxed{} must appear somewhere") that lets
    the model dump many boxes and hope the right one is among them.
    """
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    pred_boxed = extract_boxed_answers(solution_str)
    gt_boxed = extract_boxed_answers_ground_truth(ground_truth)

    # No boxed answer at all -> no format reward
    if not pred_boxed:
        return 0.0

    pred_norm = [normalize_math_expression(p) for p in pred_boxed]
    gt_norm = [normalize_math_expression(g) for g in gt_boxed]

    # MCQ: ground truth has no \\boxed{} (bare letter like "D")
    if not gt_norm:
        gt_bare = normalize_math_expression(ground_truth.strip())
        if len(pred_norm) == 1 and pred_norm[0] == gt_bare:
            return correct_score
        return format_score  # format right, answer wrong

    # Full credit: strict one-to-one positional match
    if len(pred_norm) == len(gt_norm) and all(p == g for p, g in zip(pred_norm, gt_norm)):
        return correct_score

    # Partial credit: some ground-truth answers matched (content-wise)
    matched = sum(1 for g in gt_norm if g in pred_norm)
    if matched > 0:
        return max(partial_score, format_score)
    return format_score  # format right, nothing correct

import re
import os
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import PartialState
from tqdm import tqdm

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 mine/cal_metric.py
'''

_SOLUTION_CLIP_CHARS = 500

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_boxed_answers(text: str) -> list:
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]

def normalize_math_expression(s: str) -> str:
    return re.sub(r'\s+', '', s)

def is_correct(solution_str: str, ground_truth: str) -> bool:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    pred_boxed = extract_boxed_answers(solution_str)
    gt_boxed = extract_boxed_answers(ground_truth)

    # Normalize everything upfront
    pred_norm = [normalize_math_expression(p) for p in pred_boxed]
    gt_norm = [normalize_math_expression(g) for g in gt_boxed]

    # If ground truth has no \boxed{} (e.g. MCQ bare answer like "D"),
    # compare the model's boxed answers against the bare gt directly.
    if not gt_norm:
        gt_bare = normalize_math_expression(ground_truth.strip())
        return any(p == gt_bare for p in pred_norm)

    # Every ground-truth answer must appear somewhere in the model's predictions.
    # Extra predictions are allowed (model may repeat answers while still being correct).
    for gt in gt_norm:
        if gt not in pred_norm:
            return False
    return True


state = PartialState()
local_rank = state.local_process_index
world_size = state.num_processes

if local_rank == 0:
    print(f"🚀 Using {world_size} GPUs for data‑parallel inference.")


model_path = "checkpoints/qwen_3b_sft_grponew1400"   
data_path = "data/test.parquet"

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
)
model.eval()

df = pd.read_parquet(data_path)
total_samples = len(df)
print(f"[GPU {local_rank}] Total samples: {total_samples}")


chunk_size = total_samples // world_size
start = local_rank * chunk_size
end = start + chunk_size
if local_rank == world_size - 1:
    end = total_samples
local_df = df.iloc[start:end].reset_index(drop=True)
print(f"[GPU {local_rank}] Processing samples {start}~{end-1} (total: {len(local_df)})")

BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024

prompts, ground_truths, types = [], [], []
for _, row in local_df.iterrows():
    messages = row["prompt"]
    gt = row["reward_model"]["ground_truth"]
    sample_type = row.get("extra_info", {}).get("type", "unknown")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(text)
    ground_truths.append(gt)
    types.append(sample_type)


all_responses = []
all_gen_lens = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"GPU {local_rank}", disable=local_rank != 0):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(local_rank)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # do_sample=False,          
            temperature=0.6,
            top_p=0.9,
            # repetition_penalty=1.1
        )

    for j, input_ids in enumerate(inputs.input_ids):
        response_ids = outputs[j][len(input_ids):]
        gen_len = len(response_ids)
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        all_responses.append(response)
        all_gen_lens.append(gen_len)


results_local = []
for idx, (gt, resp, typ, glen) in enumerate(zip(ground_truths, all_responses, types, all_gen_lens)):
    correct = is_correct(resp, gt)
    results_local.append({
        "global_index": start + idx,
        "type": typ,
        "ground_truth": gt,
        "response": resp,
        "correct": correct,
        "output_tokens": glen,
    })

temp_file = f"temp_results_rank_{local_rank}.csv"
pd.DataFrame(results_local).to_csv(temp_file, index=False)
print(f"[GPU {local_rank}] Saved {len(results_local)} results.")

state.wait_for_everyone()

if local_rank == 0:
    all_files = [f"temp_results_rank_{i}.csv" for i in range(world_size)]
    dfs = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(dfs, ignore_index=True)

    total = len(full_df)
    correct_total = full_df["correct"].sum()
    overall_acc = correct_total / total if total > 0 else 0
    overall_avg_tokens = full_df["output_tokens"].mean()

    type_stats = full_df.groupby("type").agg(
        correct_sum=("correct", "sum"),
        count=("correct", "count"),
        avg_tokens=("output_tokens", "mean")
    )
    type_stats["acc"] = type_stats["correct_sum"] / type_stats["count"]

    print("\n" + "=" * 80)
    print(f"✅ Overall Pass@1: {overall_acc:.4f}  ({correct_total}/{total})")
    print(f"📏 Average output tokens (overall): {overall_avg_tokens:.2f}")
    print("=" * 80)
    print("📊 Per‑type Stats:")
    for typ, row in type_stats.iterrows():
        print(f"  {typ:20s}: acc={row['acc']:.4f}  ({int(row['correct_sum'])}/{int(row['count'])})  avg_tokens={row['avg_tokens']:.2f}")

    output_file = "mine/results/evaluation_results_3b_sft_grponew1400.csv"
    with open(output_file, 'w') as f:
        f.write(f"# Overall Pass@1: {overall_acc:.4f} ({correct_total}/{total})\n")
        f.write(f"# Overall average output tokens: {overall_avg_tokens:.2f}\n")
        f.write("# Per‑type Pass@1 and average tokens:\n")
        for typ, row in type_stats.iterrows():
            f.write(f"#   {typ}: acc={row['acc']:.4f} ({int(row['correct_sum'])}/{int(row['count'])})  avg_tokens={row['avg_tokens']:.2f}\n")
        f.write("#\n")
        full_df.to_csv(f, index=False)

    print(f"\n💾 Full results saved to {output_file}")

    for f in all_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"   Removed {f}")
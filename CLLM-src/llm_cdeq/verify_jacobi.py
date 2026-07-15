from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from .config import load_config, resolve_repo_path
from .evaluate import PROMPT, load_gsm8k_questions
from .runtime import seed_everything


def greedy_ar_block(
    model: LlamaForCausalLM, prefix: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one greedy block with the same no-cache forward used by Jacobi."""
    generated = prefix
    block: list[torch.Tensor] = []
    for _ in range(block_size):
        logits = model(input_ids=generated, use_cache=False).logits[:, -1]
        token = logits.argmax(dim=-1, keepdim=True)
        block.append(token)
        generated = torch.cat((generated, token), dim=1)
    return generated, torch.cat(block, dim=1)


def vanilla_jacobi_endpoint(
    model: LlamaForCausalLM,
    prefix: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Iterate the full causal token block until it is an exact self-mapping."""
    block_size = initial.shape[1]
    prompt_length = prefix.shape[1]
    state = initial
    for iteration in range(1, block_size + 2):
        full = torch.cat((prefix, state), dim=1)
        logits = model(input_ids=full, use_cache=False).logits
        next_state = logits[
            :, prompt_length - 1 : prompt_length + block_size - 1
        ].argmax(dim=-1)
        if torch.equal(next_state, state):
            return state, iteration
        state = next_state
    raise RuntimeError("vanilla Jacobi did not converge within block_size + 1 iterations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify vanilla Jacobi and greedy AR endpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"), default="sdpa")
    parser.add_argument("--output")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    rng = random.Random(seed)
    target = resolve_repo_path(config, config["paths"]["target_model"])
    model = LlamaForCausalLM.from_pretrained(
        target,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_backend,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target, use_fast=False)
    test_path = resolve_repo_path(config, config["paths"]["gsm8k_test"])
    questions = load_gsm8k_questions(test_path, None)
    block_size = int(config["model"]["block_size"])
    checked = 0
    records: list[dict] = []
    progress = tqdm(total=args.blocks, desc="AR/Jacobi endpoint blocks")
    for question_id, question in enumerate(questions):
        prefix = tokenizer(PROMPT.format(input=question), return_tensors="pt").input_ids.to(args.device)
        block_id = 0
        while checked < args.blocks:
            ar_full, ar_endpoint = greedy_ar_block(model, prefix, block_size)
            random_point = torch.tensor(
                rng.choices(prefix[0].tolist(), k=block_size - 1),
                device=args.device,
                dtype=torch.long,
            ).view(1, -1)
            initial = torch.cat((ar_endpoint[:, :1], random_point), dim=1)
            jacobi_endpoint, iterations = vanilla_jacobi_endpoint(
                model, prefix, initial
            )
            accurate_length = block_size
            equal = bool(torch.equal(jacobi_endpoint, ar_endpoint))
            records.append(
                {
                    "question_id": question_id,
                    "block_id": block_id,
                    "equal": equal,
                    "jacobi_iterations": int(iterations),
                    "accurate_length": int(accurate_length),
                    "ar_tokens": ar_endpoint.cpu().tolist()[0],
                    "jacobi_tokens": jacobi_endpoint.cpu().tolist()[0],
                }
            )
            checked += 1
            block_id += 1
            progress.update(1)
            prefix = ar_full
            if bool(ar_endpoint.eq(tokenizer.eos_token_id).any()):
                break
        if checked >= args.blocks:
            break
    progress.close()
    matches = sum(record["equal"] for record in records)
    report = {
        "requested_blocks": args.blocks,
        "checked_blocks": checked,
        "matching_blocks": matches,
        "agreement": matches / max(checked, 1),
        "attention_backend": args.attention_backend,
        "records": records,
    }
    output = Path(args.output) if args.output else resolve_repo_path(
        config, config["paths"]["output_dir"]
    ) / "vanilla_jacobi_equivalence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if checked != args.blocks or matches != checked:
        raise RuntimeError(f"Jacobi/AR endpoint agreement failed: {matches}/{checked}")


if __name__ == "__main__":
    main()

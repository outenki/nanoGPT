"""
Evaluate model by number agreement.
Input: a dataset with sentences and a model.
Output: a dataset with sentences and the model's predictions.
"""

import torch
import argparse
import json
from pathlib import Path
import tqdm

from transformers import GPT2Tokenizer
from model import GPTConfig, GPT


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', '-mp', dest='model_path', type=str,
        help='Model path to load from. (pt)'
    )
    parser.add_argument(
        '--val-data', '-vd', dest='data_name', type=str,
        help='Path to test data. (json)'
    )
    parser.add_argument(
        '-mps', dest='mps', action='store_true',
        help="Use MPS device for evaluation. (MacOS GPU)"
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save results.'
    )
    return parser.parse_args()


def load_model(ckpt_path, device) -> GPT:
    """
    Load the model from the checkpoint path.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in state_dict:
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model


def score_candidate(prompt, continuation, tokenizer, model) -> float:
    # Compute log-likelihood of a candidate continuation given a prompt
    input_text = prompt + continuation
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

    input_ids = input_ids.to(next(model.parameters()).device)
    prompt_ids = prompt_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        _, loss = model(input_ids, input_ids)
        total_log_prob = -loss.item() * (input_ids.size(1) - prompt_ids.size(1))
    return total_log_prob


def score_candidates(prompt, option1, option2, tokenizer, model) -> tuple[float, float]:
    """
    Score two candidates given a prompt.
    Returns the scores for both candidates.
    """
    score1 = score_candidate(prompt, option1, tokenizer, model)
    score2 = score_candidate(prompt, option2, tokenizer, model)
    return score1, score2


def score_samples(samples, tokenizer, model) -> list[dict]:
    # Score a list of samples with prompts and two options.
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        score1, score2 = score_candidates(prompt, option1, option2, tokenizer, model)
        samples[i]["score1"] = score1
        samples[i]["score2"] = score2
    return samples


def analyze_results(samples) -> dict:
    """
    Analyze the results of the evaluation.
    Returns a summary of the scores.
    """
    correct = 0
    total = 0
    for sample in samples:
        predicted = sample["option1"] if sample["score1"] > sample["score2"] else sample["option2"]
        is_correct = (predicted == sample["answer"])
        correct += int(is_correct)
        total += 1
    accuracy = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    args = read_args()

    # ======== Check arguments ========
    ckpt_path = args.model_path
    eval_data_path = args.data_name
    out_path = args.out_path
    if not ckpt_path or not eval_data_path or not out_path:
        raise ValueError("Please provide model path, evaluation data path, and output path.")
    if Path(ckpt_path).suffix != '.pt':
        raise ValueError("Model path must be a .pt file.")
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path, device)
    model.to(device)
    model.eval()

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in tqdm.tqdm(f.readlines(), desc="Loading evaluation data"):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict) or "prompt" not in sample or "option1" not in sample or "option2" not in sample:
                    raise ValueError("Each line must be a JSON object with 'prompt', 'option1', and 'option2'.")
                eval_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    if not isinstance(eval_samples, list):
        raise ValueError("Evaluation data must be a list of samples.")

    # ========= Score samples ========
    eval_samples = score_samples(eval_samples, tokenizer, model)
    results = analyze_results(eval_samples)

    # ========= Save results ========
    out_file = Path(out_path) / "evaluated_samples.json"
    print(f"Saving evaluation results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(eval_samples, f, indent=4)

    out_file = Path(out_path) / "evaluation_summary.json"
    print(f"Saving summary results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

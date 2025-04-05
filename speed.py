import json
import argparse
from transformers import AutoTokenizer
import numpy as np


def speed(jsonl_file, jsonl_file_base, tokenizer, task=None, report=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def load_and_filter(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if task is None or task == "overall":
                    data.append(j)
                elif j["category"] == task:
                    data.append(j)
        return data

    data = load_and_filter(jsonl_file)
    speeds = []
    for dp in data:
        output = dp["output"]
        latency = dp["latency"]
        tokens = len(tokenizer(output).input_ids) - 1
        speeds.append(tokens / latency)

    data_base = load_and_filter(jsonl_file_base)
    speeds_base = []
    for dp in data_base:
        output = dp["output"]
        latency = dp["latency"]
        tokens = len(tokenizer(output).input_ids) - 1
        speeds_base.append(tokens / latency)

    tps = np.mean(speeds)
    tps_base = np.mean(speeds_base)
    ratio = tps / tps_base

    if report:
        print("=" * 30, "Task: ", task, "=" * 30)
        print(f"Tokens per second: {tps:.2f}")
        print(f"Tokens per second for the baseline: {tps_base:.2f}")
        print(f"Speedup ratio: {ratio:.3f}")
    return tps, tps_base, ratio


def get_all_categories(jsonl_file, jsonl_file_base, tokenizer_path):
    categories = set()
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "category" in j:
                categories.add(j["category"])

    categories = sorted(categories)
    categories.append("overall")

    for cat in categories:
        speed(jsonl_file, jsonl_file_base, tokenizer_path, task=cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True, help="Path to speculative decoding results (.jsonl)")
    parser.add_argument("--base-path", type=str, required=True, help="Path to baseline results (.jsonl)")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer path or HuggingFace model ID")
    args = parser.parse_args()

    get_all_categories(args.file_path, args.base_path, args.tokenizer_path)

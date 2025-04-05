import argparse
import json
from collections import defaultdict

def load_answers(filepath):
    """Load jsonl and return dict: qid -> (output, category)"""
    qid_to_output = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            qid = obj['question_id']
            qid_to_output[qid] = {
                'output': obj['output'].strip(),
                'category': obj['category']
            }
    return qid_to_output

def compare_outputs(jsonfile1, jsonfile2):
    """Compare outputs and group by task (category)"""
    ans1 = load_answers(jsonfile1)
    ans2 = load_answers(jsonfile2)

    stats = defaultdict(lambda: {'total': 0, 'match': 0})

    for qid in ans1:
        if qid not in ans2:
            print(f"Warning: question_id {qid} not found in second file.")
            continue
        cat = ans1[qid]['category']
        out1 = ans1[qid]['output']
        out2 = ans2[qid]['output']
        stats[cat]['total'] += 1
        if out1 == out2:
            stats[cat]['match'] += 1

    # Overall stats
    total = sum(v['total'] for v in stats.values())
    match = sum(v['match'] for v in stats.values())

    for cat, val in sorted(stats.items()):
        acc = val['match'] / val['total'] * 100 if val['total'] > 0 else 0
        print(f"============= Task: {cat} =============")
        print(f"Total Questions: {val['total']}")
        print(f"Exact Matches: {val['match']}")
        print(f"Accuracy: {acc:.2f}%\n")

    overall_acc = match / total * 100 if total > 0 else 0
    print(f"============= Overall =============")
    print(f"Total Questions: {total}")
    print(f"Exact Matches: {match}")
    print(f"Accuracy: {overall_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        required=True,
        help="Folder path containing both result files (optional if absolute paths used)."
    )
    parser.add_argument(
        "--jsonfile1",
        required=True,
        help="First result file (e.g. baseline)."
    )
    parser.add_argument(
        "--jsonfile2",
        required=True,
        help="Second result file (e.g. speculative decoding)."
    )
    args = parser.parse_args()

    file1 = args.jsonfile1
    file2 = args.jsonfile2
    if args.file_path:
        file1 = f"{args.file_path.rstrip('/')}/{args.jsonfile1}"
        file2 = f"{args.file_path.rstrip('/')}/{args.jsonfile2}"

    compare_outputs(file1, file2)

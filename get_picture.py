import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

file_path = "expert_routing_ngram.csv"

with open(file_path, 'r') as f:
    lines = f.read().splitlines()

layer_expert_map = defaultdict(list)
current_layer = None

for line in lines:
    if ',' not in line:
        current_layer = int(line)
    else:
        experts = [int(e.strip()) for e in line.split(',')]
        layer_expert_map[current_layer].extend(experts)

df = pd.DataFrame([
    {'layer': layer, 'expert': expert}
    for layer, experts in layer_expert_map.items()
    for expert in experts
])

count_df_ngram = df.groupby(['layer', 'expert']).size().unstack(fill_value=0).sort_index()

vmax = count_df_ngram.values.max()
vmin = 0

plt.figure(figsize=(16, 6))
plt.imshow(count_df_ngram, aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax)
plt.colorbar(label='Expert Frequency')
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title("Expert Usage Distribution per Layer (with ngram)")
plt.xticks(ticks=range(len(count_df_ngram.columns)), labels=count_df_ngram.columns, rotation=90)
plt.yticks(ticks=range(len(count_df_ngram.index)), labels=count_df_ngram.index)
plt.tight_layout()
plt.show()


file_path = "expert_routing_nongram.csv"

with open(file_path, 'r') as f:
    lines = f.read().splitlines()

layer_expert_map = defaultdict(list)
current_layer = None

for line in lines:
    if ',' not in line:
        current_layer = int(line)
    else:
        experts = [int(e.strip()) for e in line.split(',')]
        layer_expert_map[current_layer].extend(experts)

df = pd.DataFrame([
    {'layer': layer, 'expert': expert}
    for layer, experts in layer_expert_map.items()
    for expert in experts
])

count_df_nongram = df.groupby(['layer', 'expert']).size().unstack(fill_value=0).sort_index()

plt.figure(figsize=(16, 6))
plt.imshow(count_df_nongram, aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax)
plt.colorbar(label='Expert Frequency')
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title("Expert Usage Distribution per Layer (without ngram)")
plt.xticks(ticks=range(len(count_df_nongram.columns)), labels=count_df_nongram.columns, rotation=90)
plt.yticks(ticks=range(len(count_df_nongram.index)), labels=count_df_nongram.index)
plt.tight_layout()
plt.show()


count_df_ngram, count_df_nongram = count_df_ngram.align(count_df_nongram, fill_value=0)

diff_df = count_df_ngram - count_df_nongram

plt.figure(figsize=(16, 6))
plt.imshow(count_df_nongram, aspect='auto', cmap='Blues')
plt.colorbar(label='Expert Frequency')
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title("Expert Usage Distribution Difference per Layer")
plt.xticks(ticks=range(len(diff_df.columns)), labels=diff_df.columns, rotation=90)
plt.yticks(ticks=range(len(diff_df.index)), labels=diff_df.index)
plt.tight_layout()
plt.show()
plt.savefig("expert_usage_difference_heatmap.png", dpi=300)
plt.close()

count_df_ngram, count_df_nongram = count_df_ngram.align(count_df_nongram, fill_value=0)

diff_df = count_df_ngram - count_df_nongram

plt.figure(figsize=(16, 6))
plt.imshow(count_df_nongram, aspect='auto', cmap='Blues')
plt.colorbar(label='Expert Frequency')
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title("Expert Usage Distribution Difference per Layer")
plt.xticks(ticks=range(len(diff_df.columns)), labels=diff_df.columns, rotation=90)
plt.yticks(ticks=range(len(diff_df.index)), labels=diff_df.index)
plt.tight_layout()
plt.show()
plt.savefig("expert_usage_difference_heatmap.png", dpi=300)
plt.close()


ratio_df = count_df_ngram / count_df_nongram
plt.figure(figsize=(16, 6))
plt.imshow(ratio_df, aspect='auto', cmap='Blues')
plt.colorbar(label='Expert Frequency')
plt.xlabel("Expert ID")
plt.ylabel("Layer")
plt.title("Expert Usage Distribution Ratio per Layer")
plt.xticks(ticks=range(len(ratio_df.columns)), labels=ratio_df.columns, rotation=90)
plt.yticks(ticks=range(len(ratio_df.index)), labels=ratio_df.index)
plt.tight_layout()
plt.show()
plt.savefig("expert_usage_ratio_heatmap.png", dpi=300)
plt.close()
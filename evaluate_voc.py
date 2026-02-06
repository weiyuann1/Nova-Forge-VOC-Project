import json
import re
import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def parse_json_output(text):
    try:
        if isinstance(text, str):
            if text.startswith("['") and text.endswith("']"):
                text = text[2:-2].replace("\\'", "'")
        if isinstance(text, list) and len(text) > 0:
            text = text[0]
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            return json.loads(text[start_idx:end_idx]), 'json'
    except:
        pass
    return None, None

def extract_labels_regex(text, level):
    if isinstance(text, str):
        if text.startswith("['") and text.endswith("']"):
            text = text[2:-2]
    if isinstance(text, list) and len(text) > 0:
        text = text[0]
    
    pattern = f'"{level}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"'
    matches = re.findall(pattern, text)
    
    labels = []
    for match in matches:
        label = match.replace('\\"', '"').replace("\\'", "'")
        if level == 'E4' and ':' in label:
            label = label.split(':', 1)[0].strip()
        labels.append(label)
    return labels

def extract_labels(text, level):
    parsed, method = parse_json_output(text)
    if parsed is not None:
        labels = []
        seen = set()
        for item in parsed:
            if level in item:
                label = item[level]
                if level == 'E4' and ':' in label:
                    label = label.split(':', 1)[0].strip()
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)
                    if len(labels) >= 10:
                        break
        return labels, method
    else:
        labels = extract_labels_regex(text, level)
        seen = set()
        deduped = []
        for label in labels:
            if label and label not in seen:
                seen.add(label)
                deduped.append(label)
                if len(deduped) >= 10:
                    break
        return deduped, 'regex'

input_file = sys.argv[1] if len(sys.argv) > 1 else 'inference_output.jsonl'
output_dir = sys.argv[2] if len(sys.argv) > 2 else './'

print(f'Input: {input_file}')
print(f'Output: {output_dir}')

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

json_count = sum(1 for s in data if extract_labels(s['inference'], 'E1')[1] == 'json')
regex_count = len(data) - json_count
print(f'Parsing: JSON={json_count}, Regex={regex_count}')

results = []
for level in ['E1', 'E2', 'E3', 'E4']:
    y_true_all, y_pred_all = [], []
    
    for sample in data:
        gt_labels, _ = extract_labels(sample['gold'], level)
        pred_labels, _ = extract_labels(sample['inference'], level)
        
        y_true_all.extend(gt_labels)
        y_pred_all.extend(pred_labels[:len(gt_labels)])
        if len(pred_labels) < len(gt_labels):
            y_pred_all.extend([''] * (len(gt_labels) - len(pred_labels)))
    
    if not y_true_all:
        continue
    
    all_labels = sorted(list(set(y_true_all + y_pred_all) - {''}))
    
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=all_labels, average='weighted', zero_division=0
    )
    
    results.append({'level': level, 'precision': weighted_p, 'recall': weighted_r, 'f1': weighted_f1})
    print(f"{level}: P={weighted_p:.4f}, R={weighted_r:.4f}, F1={weighted_f1:.4f}")

rows = []
for idx, sample in enumerate(data, 1):
    gold_e1, _ = extract_labels(sample['gold'], 'E1')
    pred_e1, _ = extract_labels(sample['inference'], 'E1')
    gold_e2, _ = extract_labels(sample['gold'], 'E2')
    pred_e2, _ = extract_labels(sample['inference'], 'E2')
    gold_e3, _ = extract_labels(sample['gold'], 'E3')
    pred_e3, _ = extract_labels(sample['inference'], 'E3')
    gold_e4, _ = extract_labels(sample['gold'], 'E4')
    pred_e4, _ = extract_labels(sample['inference'], 'E4')
    
    max_len = max(len(gold_e1), len(pred_e1))
    
    for i in range(max_len):
        rows.append({
            'Sample_ID': idx,
            'Gold_E1': gold_e1[i] if i < len(gold_e1) else '',
            'Pred_E1': pred_e1[i] if i < len(pred_e1) else '',
            'Gold_E2': gold_e2[i] if i < len(gold_e2) else '',
            'Pred_E2': pred_e2[i] if i < len(pred_e2) else '',
            'Gold_E3': gold_e3[i] if i < len(gold_e3) else '',
            'Pred_E3': pred_e3[i] if i < len(pred_e3) else '',
            'Gold_E4': gold_e4[i] if i < len(gold_e4) else '',
            'Pred_E4': pred_e4[i] if i < len(pred_e4) else ''
        })

df = pd.DataFrame(rows)
excel_file = f'{output_dir}/evaluation_final.xlsx'
df.to_excel(excel_file, index=False)
print(f'\nExcel: {len(rows)} rows -> {excel_file}')

json_file = f'{output_dir}/evaluation_results.json'
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'JSON -> {json_file}')

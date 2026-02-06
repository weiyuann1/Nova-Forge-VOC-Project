# VOC-Nova-Forge

This folder contains all scripts and configuration files required to run VOC (Voice of Customer) classification tasks on Amazon SageMaker HyperPod.

## File Description

### 1. Training Files

#### `nova_lite_2_0_p5_gpu_sft_data_mixing_voc.yaml`
- **Purpose**: HyperPod SFT (Supervised Fine-Tuning) training configuration file with VOC data mixing
- **Key Configuration**:
  - Model: `amazon.nova-2-lite-v1:0:256k`
  - Instance count: 4
  - Training steps: 500
  - Sequence length: 32768
  - Global batch size: 32
  - Data mixing ratio: 75% customer data + 25% Nova data
- **Data Mixing Strategy**:
  - customer_data: 75%
  - nova_data includes multiple categories: agents, baseline, chat, code, reasoning, etc.

#### `example.jsonl`
- **Purpose**: VOC training dataset (Bedrock format)
- **Format**: JSONL format, one training sample per line
- **Content**: Multi-level classification annotations for user comments (L1-L4 hierarchy)

### 2. Evaluation Files
#### `nova_lite_2_0_p5_48xl_gpu_bring_your_own_dataset_eval.yaml`
- **Purpose**: HyperPod evaluation task configuration file
- **Key Configuration**:
  - Model: `amazon.nova-2-lite-v1:0:256k`
  - Instance count: 1
  - Evaluation task: gen_qa
  - Inference parameters: max_new_tokens=8196, temperature=0

#### `evaluate_voc.py`
- **Purpose**: VOC classification result evaluation script
- **Features**:
  - Parse inference output in JSON format
  - Extract classification labels for E1-E4 hierarchy levels
  - Calculate Precision, Recall, F1-Score metrics
  - Generate detailed comparison results in Excel format
  - Output evaluation results in JSON format
- **Input**: inference_output.jsonl (inference result file)
- **Output**: 
  - evaluation_final.xlsx (detailed comparison table)
  - evaluation_results.json (evaluation metrics)

## Usage Workflow
### 1. Training Phase
Start training job using HyperPod CLI:

```bash
hyperpod start-job \
-n kubeflow \
--recipe fine-tuning/nova/nova_2_0/nova_lite/SFT/nova_lite_2_0_p5_gpu_sft \
--override-parameters '{
"instance_type": "ml.p5.48xlarge",
"recipes.run.name": "nova-voc-datamixing",
"recipes.run.replicas": 4,
"container": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-SFT-V2-NO-MERGE-latest",
"recipes.run.data_s3_path": $train_data,
"recipes.run.output_s3_path": $output_dir,
"recipes.training_config.save_steps": 264,
"recipes.training_config.max_steps": 264,
"recipes.training_config.save_top_k": 1,
"recipes.training_config.max_length": 8192,
"recipes.training_config.global_batch_size": 32,
"recipes.training_config.reasoning_enabled": false
}'
```

mix nova data
```bash
hyperpod start-job \
-n kubeflow \
--recipe fine-tuning/nova/nova_2_0/nova_lite/SFT/nova_lite_2_0_p5_gpu_sft_data_mixing_voc \
--override-parameters '{
"instance_type": "ml.p5.48xlarge",
"recipes.run.name": "nova-sft-voc-datamixing",
"recipes.run.replicas": 4,
"container": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-HP-SFT-V2-DATAMIXING-latest",
"recipes.run.data_s3_path": $train_data,
"recipes.run.output_s3_path": $output_dir,
"recipes.training_config.max_steps": 500,
"recipes.training_config.save_top_k": 1,
"recipes.training_config.max_length": 32768,
"recipes.training_config.global_batch_size": 32,
"recipes.training_config.reasoning_enabled": false,
"recipes.data_mixing.sources.customer_data.percent": 75
}'
```


### 2. Inference Phase

After training completes, run inference using evaluation configuration:

```bash
hyperpod start-job -n kubeflow \
--recipe evaluation/nova/nova_2_0/nova_lite/nova_lite_2_0_p5_48xl_gpu_bring_your_own_dataset_eval \
--override-parameters '{
"instance_type": "p5.48xlarge",
"container": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-evaluation-repo:SM-HP-Eval-Beta-latest",
"recipes.run.name": "sql-sft-evaluate",
"recipes.run.model_name_or_path": "",
"recipes.run.output_s3_path": "",
"recipes.run.data_s3_path": ""
}'
```

### 3. Evaluation Phase

After inference completes, run evaluation script:

```bash
# Evaluate using evaluate_voc.py
python evaluate_voc.py <inference_output.jsonl> <output_dir>
```

## Important Notes

1. Training data must be uploaded to S3 and the correct path specified in the configuration file
2. Output path must also be configured as a valid S3 path
3. Ensure HyperPod cluster has sufficient resources (p5.48xlarge instances)
4. Evaluation script requires sklearn and pandas dependencies
5. Inference output format must be JSONL, with each line containing gold and inference fields


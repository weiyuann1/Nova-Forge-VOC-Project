# VOC-Nova-Forge

这个文件夹包含了在Amazon SageMaker HyperPod上运行VOC (Voice of Customer) 分类任务所需的所有脚本和配置文件。

## 文件说明

### 1. 训练相关文件

#### `nova_lite_2_0_p5_gpu_sft_data_mixing_voc.yaml`
- **用途**: HyperPod SFT (Supervised Fine-Tuning) 训练配置文件，使用VOC数据混合
- **关键配置**:
  - 模型: `amazon.nova-2-lite-v1:0:256k`
  - 实例数: 4个
  - 训练步数: 500 
  - 序列长度: 32768
  - 全局批次大小: 32
  - 数据混合比例: 75% 客户数据 + 25% Nova数据
- **数据混合策略**:
  - customer_data: 75%
  - nova_data包含多个类别: agents, baseline, chat, code, reasoning等

#### `train_bedrock_in_domin.jsonl`
- **用途**: VOC训练数据集 (Bedrock格式)
- **格式**: JSONL格式，每行一个训练样本
- **内容**: 用户评论的多级分类标注数据 (L1-L4层级)

### 2. 评估相关文件
#### `nova_lite_2_0_p5_48xl_gpu_bring_your_own_dataset_eval.yaml`
- **用途**: HyperPod评估任务配置文件
- **关键配置**:
  - 模型: `amazon.nova-2-lite-v1:0:256k`
  - 实例数: 1
  - 评估任务: gen_qa
  - 推理参数: max_new_tokens=8196, temperature=0

#### `evaluate_voc.py`
- **用途**: VOC分类结果评估脚本
- **功能**:
  - 解析推理输出的JSON格式
  - 提取E1-E4四个层级的分类标签
  - 计算Precision, Recall, F1-Score指标
  - 生成Excel格式的详细对比结果
  - 输出JSON格式的评估结果
- **输入**: inference_output.jsonl (推理结果文件)
- **输出**: 
  - evaluation_final.xlsx (详细对比表格)
  - evaluation_results.json (评估指标)

## 使用流程
### 1. 训练阶段
使用HyperPod CLI启动训练任务:

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


### 2. 推理阶段

训练完成后，使用评估配置进行推理:

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

### 3. 评估阶段

推理完成后，运行评估脚本:

```bash
# 使用evaluate_voc.py评估
python evaluate_voc.py <inference_output.jsonl> <output_dir>
```

## 注意事项

1. 训练数据需要上传到S3，并在配置文件中指定正确的路径
2. 输出路径也需要配置为有效的S3路径
3. 确保HyperPod集群有足够的资源 (p5.48xlarge实例)
4. 评估脚本需要sklearn和pandas依赖库
5. 推理输出格式必须是JSONL，每行包含gold和inference字段


# Functional transcription of the Gemma SFT script supplied by the user on 2026-09-22.
# Original settings and resume behavior retained; comments/console banners omitted.
# Archive/reference only. Use scripts/train_warm_sft.py for a new independent run.
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor, AutoModelForMultimodalLM

os.environ['HF_HOME']='/workspace/warm_models/hf_cache'
os.environ['HF_DATASETS_CACHE']='/workspace/warm_models/hf_cache/datasets'
os.environ['TRANSFORMERS_CACHE']='/workspace/warm_models/hf_cache/models'
MODEL_ID='google/gemma-4-12b-it'
OUTPUT_DIR='/workspace/warm_models/lora_warm_adapter_gemma_12b'
DATA_PATH='data/dataset_warm_final.json'

processor=AutoProcessor.from_pretrained(MODEL_ID,trust_remote_code=True)
model=AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True)
if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
dataset=load_dataset('json',data_files=DATA_PATH,split='train')


def format_with_template(example):
    messages=example.get('messages',example.get('conversations',[]))
    standard_messages=[]
    for msg in messages:
        role=str(msg.get('role',msg.get('from',''))).strip().lower()
        if role in ['human','prompter']:role='user'
        elif role in ['gpt','bot','model']:role='assistant'
        standard_messages.append({'role':role,'content':msg.get('content',msg.get('value',''))})
    return {'text':processor.apply_chat_template(standard_messages,tokenize=False,add_generation_prompt=False)}


dataset=dataset.map(format_with_template,remove_columns=dataset.column_names)


def get_mm_linear_modules(model):
    linear_modules=set()
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            exclude_keywords=['lm_head','vision','audio','multi_modal_projector']
            if any(keyword in name.lower() for keyword in exclude_keywords):continue
            names=name.split('.')
            linear_modules.add(names[0] if len(names)==1 else names[-1])
    return list(linear_modules)


target_modules=get_mm_linear_modules(model)
lora_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',
                       task_type='CAUSAL_LM',target_modules=target_modules)
training_args=SFTConfig(
    output_dir=OUTPUT_DIR,dataset_text_field='text',num_train_epochs=10,
    per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=1e-5,lr_scheduler_type='cosine',warmup_steps=100,
    logging_steps=1,save_strategy='steps',save_steps=200,save_total_limit=3,
    bf16=True,optim='adamw_torch_fused',gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},report_to='none',max_length=2048)
trainer=SFTTrainer(model=model,train_dataset=dataset,args=training_args,
                   peft_config=lora_config,processing_class=processor)
trainer.train(resume_from_checkpoint=True)
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

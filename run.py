import argparse, json, random, re, string, csv, gc, hashlib
from pathlib import Path
from collections import Counter

# 0: target single prompt; 1..4: seen pool; 5..8: unseen evaluation pool.
PROMPTS = [
'Context:\n{c}\n\nQuestion: {q}\nReturn only the shortest answer span copied from the context.',
'Read the passage and extract the answer. Output only the answer text.\nPassage: {c}\nQuery: {q}',
'Find the exact words in the text that answer the question. No explanation.\nText: {c}\nQuestion: {q}',
'Extract a short answer from the evidence below. Respond with the answer alone.\nEvidence:\n{c}\nAsked: {q}',
'Use this source to answer the query with a verbatim span only.\nSource: {c}\nQuery: {q}',
'Which short phrase in the document answers the question? Write that phrase alone.\nDocument: {c}\nQuestion: {q}',
'Answer the following using an exact excerpt from the passage. Include nothing else.\nQuestion: {q}\nPassage: {c}',
'Locate the answer to the inquiry in the supplied material. Return only the matching words.\nInquiry: {q}\nMaterial: {c}',
'Provide a concise verbatim answer to this question based on the reading. No commentary.\nReading: {c}\nQuestion: {q}'
]
SYSTEM = 'You answer extractive reading comprehension questions.'

def normalize(s):
    s = s.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def score(pred, answers):
    p = normalize(pred).split()
    def f(a):
        g = normalize(a).split(); n = sum((Counter(p) & Counter(g)).values())
        return (2*n/(len(p)+len(g))) if p and g else float(p == g)
    return max(float(normalize(pred)==normalize(a)) for a in answers), max(f(a) for a in answers)

def prompt(tok, row, pid):
    return tok.apply_chat_template([{'role':'system','content':SYSTEM}, {'role':'user','content':PROMPTS[pid].format(c=row['context'],q=row['question'])}], tokenize=False, add_generation_prompt=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--train-n',type=int,default=2000); ap.add_argument('--test-n',type=int,default=100)
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--seeds',type=int,nargs='+',default=[13,42,71])
    ap.add_argument('--max-length',type=int,default=1024); ap.add_argument('--out',default='results')
    ap.add_argument('--prepare-only',action='store_true'); args=ap.parse_args()
    import torch, numpy as np
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model); tok.pad_token=tok.eos_token
    raw=load_dataset('squad')
    # Held-out official validation; additionally reject identical contexts/questions.
    val_contexts={r['context'] for r in raw['validation']}; val_questions={r['question'] for r in raw['validation']}
    def fits(r):
        if not r['answers']['text']: return False
        return all(len(tok(prompt(tok,r,p),add_special_tokens=False)['input_ids'])+len(tok(r['answers']['text'][0],add_special_tokens=False)['input_ids'])+1 <= args.max_length for p in range(len(PROMPTS)))
    def select(split,n,train=False):
        chosen=[]
        for r in split.shuffle(seed=2026):
            if train and (r['context'] in val_contexts or r['question'] in val_questions): continue
            if fits(r): chosen.append(dict(r))
            if len(chosen)==n: return chosen
        raise ValueError('Not enough eligible examples')
    train=select(raw['train'],args.train_n,True); test=select(raw['validation'],args.test_n)
    (out/'data.json').write_text(json.dumps({'train':train,'test':test},ensure_ascii=False))
    (out/'config.json').write_text(json.dumps(vars(args)|{'prompts':PROMPTS,'train_ids':[r['id'] for r in train],'test_ids':[r['id'] for r in test]},indent=2))
    if args.prepare_only: return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for tuning. Use --prepare-only for data audit.')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    def load():
        return AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype).cuda()
    def collate(rows):
        m=max(len(r['input_ids']) for r in rows)
        return {k:torch.tensor([r[k]+[(-100 if k=='labels' else (tok.pad_token_id if k=='input_ids' else 0))]*(m-len(r[k])) for r in rows]) for k in ['input_ids','attention_mask','labels']}
    metrics=[]
    def evaluate(model,condition,seed):
        model.eval(); model.config.use_cache=True
        predictions=[]
        for row in test:
            for pid in range(9):
                inp=tok(prompt(tok,row,pid),return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode(): gen=model.generate(**inp,max_new_tokens=64,do_sample=False,pad_token_id=tok.pad_token_id)
                pred=tok.decode(gen[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
                em,f1=score(pred,row['answers']['text'])
                predictions.append({'id':row['id'],'pid':pid,'prediction':pred,'answers':row['answers']['text'],'em':em,'f1':f1})
        (out/f'pred_{condition}_{seed}.json').write_text(json.dumps(predictions,ensure_ascii=False))
        groups={'single_target':[0],'diverse_seen':list(range(1,5)),'diverse_unseen':list(range(5,9))}
        for name,pids in groups.items():
            # Cluster bootstrap over questions, not prompt variants.
            ems=np.array([np.mean([p['em'] for p in predictions if p['id']==r['id'] and p['pid'] in pids]) for r in test])
            rng=np.random.default_rng(2026)
            boot=np.array([rng.choice(ems,len(ems),replace=True).mean() for _ in range(2000)])
            sub=[p for p in predictions if p['pid'] in pids]
            metrics.append({'condition':condition,'seed':seed,'test':name,'em':float(ems.mean()),'f1':float(np.mean([p['f1'] for p in sub])),'em_ci_low':float(np.quantile(boot,.025)),'em_ci_high':float(np.quantile(boot,.975))})
        for pid in range(9):
            sub=[p for p in predictions if p['pid']==pid]
            metrics.append({'condition':condition,'seed':seed,'test':f'prompt_{pid}','em':float(np.mean([p['em'] for p in sub])),'f1':float(np.mean([p['f1'] for p in sub]))})
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    model=load(); evaluate(model,'base',0); del model; gc.collect(); torch.cuda.empty_cache()
    for seed in args.seeds:
        # Same QA order, optimizer steps, and answer tokens across all conditions.
        for condition in ['single_target','single_other','diverse_other','diverse_including_target']:
            set_seed(seed); model=load()
            model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',task_type='CAUSAL_LM'))
            rng=random.Random(seed)
            pool={'single_target':[0],'single_other':[1],'diverse_other':[1,2,3,4],'diverse_including_target':[0,1,2,3]}[condition]
            pids=[pool[i%len(pool)] for i in range(len(train))]; rng.shuffle(pids)
            rows=[]
            for row,pid in zip(train,pids):
                x=tok(prompt(tok,row,pid),add_special_tokens=False)['input_ids']
                y=tok(row['answers']['text'][0],add_special_tokens=False)['input_ids']+[tok.eos_token_id]
                rows.append({'input_ids':x+y,'attention_mask':[1]*(len(x)+len(y)),'labels':[-100]*len(x)+y})
            dest=out/f'{condition}_{seed}'
            config=TrainingArguments(output_dir=str(dest),num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,gradient_accumulation_steps=8,learning_rate=2e-4,warmup_ratio=.03,weight_decay=.01,logging_steps=25,save_strategy='no',report_to='none',seed=seed,data_seed=seed,bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
            model.config.use_cache=False
            Trainer(model=model,args=config,train_dataset=Dataset.from_list(rows),data_collator=collate).train()
            model.save_pretrained(dest); tok.save_pretrained(dest)
            evaluate(model,condition,seed)
            del model; gc.collect(); torch.cuda.empty_cache()
    keys=['condition','seed','test','em','f1','em_ci_low','em_ci_high']
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(metrics)
    summary=[]
    for condition in sorted({m['condition'] for m in metrics}):
        for name in ['single_target','diverse_seen','diverse_unseen']:
            ms=[m for m in metrics if m['condition']==condition and m['test']==name]
            summary.append({'condition':condition,'test':name,'em_mean':float(np.mean([m['em'] for m in ms])),'em_seed_std':float(np.std([m['em'] for m in ms],ddof=1)) if len(ms)>1 else None,'f1_mean':float(np.mean([m['f1'] for m in ms]))})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Ovis2.5 native evaluation of BLINK Spatial Relation + Relative Depth val."""
import sys as _protocol_sys
from pathlib import Path as _ProtocolPath
_protocol_sys.path.insert(0,str(_ProtocolPath(__file__).resolve().parent.parent))
from protocol import install_ovis
import argparse
import base64
import collections
import io
import json
import os
from pathlib import Path
import re
import time

import eval_ovis_native as common

SUFFIX = '\nAnswer with only the option letter in the format Answer: A or Answer: B.'


def parse_answer(response):
    # Do not use intermediate answers inside a thinking block.
    final = response.rsplit('</think>', 1)[-1].strip()
    hits = re.findall(r'(?:Final\s+answer|Answer)\s*:\s*\(?([AB])\)?\b', final, re.I)
    if hits:
        return hits[-1].upper()
    match = re.fullmatch(r'\(?([AB])\)?[.\s]*', final, re.I)
    return match.group(1).upper() if match else None


def read_data(path):
    records = []
    seen = set()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            assert row['idx'] not in seen
            seen.add(row['idx'])
            assert row['task'] in ('Spatial_Relation', 'Relative_Depth')
            assert row['answer'] in ('(A)', '(B)')
            records.append(row)
    counts = dict(collections.Counter(x['task'] for x in records))
    assert counts == {'Spatial_Relation':143,'Relative_Depth':124}, counts
    return records


def score(rows, total):
    def stats(items):
        return dict(n=len(items), correct=sum(r['correct'] for r in items),
                    accuracy_percent=100*sum(r['correct'] for r in items)/len(items) if items else None)
    values = list(rows.values())
    cats = {k:stats([x for x in values if x['task']==k]) for k in ('Spatial_Relation','Relative_Depth')}
    return dict(benchmark='BLINK two selected val subsets; not full BLINK', repeat=1,
                expected_rows=total, processed_rows=len(values), complete=len(values)==total,
                unparsed_rows=sum(r['prediction'] is None for r in values),
                subsets=cats, selected_subsets_micro_average=stats(values),
                selected_subsets_macro_accuracy_percent=sum(v['accuracy_percent'] for v in cats.values())/2
                  if all(v['n'] for v in cats.values()) else None,
                scoring='local exact option-letter match; unparsed answers are wrong; no LLM judge')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='/root/benchmark-data/BLINK/prepared/blink_val267.jsonl')
    p.add_argument('--model',default='/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B')
    p.add_argument('--output',default='/root/autodl-fs/baselines/cost_bench/ovis_compare_greedy_v1/runs/blink_native')
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--thinking-budget',type=int,default=1536)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit-per-task',type=int,default=0)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--score-only',action='store_true')
    a=p.parse_args()
    if a.limit_per_task < 0 or a.max_new_tokens <= a.thinking_budget+25:
        p.error('Invalid limit/token budget')
    data_path=Path(a.data); data=read_data(data_path)
    manifest=json.loads(data_path.with_suffix('.manifest.json').read_text())
    assert common.digest(data_path)==manifest['normalized_sha256'], 'Prepared data hash mismatch'
    if a.check_only:
        from PIL import Image
        import hashlib
        for row in data:
            raw=base64.b64decode(row['image_b64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
            with Image.open(io.BytesIO(raw)) as im: im.verify()
        print('CHECK PASSED: Spatial_Relation=143, Relative_Depth=124; all 267 images verified. No GPU/model used.')
        return
    if a.limit_per_task:
        counts=collections.Counter(); chosen=[]
        for row in data:
            if counts[row['task']] < a.limit_per_task:
                counts[row['task']]+=1;chosen.append(row)
        data=chosen
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=dict(benchmark='BLINK val Spatial_Relation + Relative_Depth',repeat=1,
                data_sha256=common.digest(data_path),script_sha256=common.digest(__file__),
                common_script_sha256=common.digest(common.__file__),
                model=str(Path(a.model).resolve()),model_config_sha256=common.digest(Path(a.model)/'config.json'),
                max_new_tokens=a.max_new_tokens,thinking_budget=a.thinking_budget,enable_thinking=True,
                seed=a.seed,do_sample=False,limit_per_task=a.limit_per_task,prompt_suffix=SUFFIX,
                prompt_policy='dataset original prompt plus output-format suffix; no extra CoT system prompt',
                scoring='local exact match; no LLM judge')
    mp=out/'blink.meta.json'
    if mp.exists():
        if json.loads(mp.read_text())!=config:
            raise RuntimeError('Code/data/settings changed; use a NEW --output directory.')
    elif a.score_only:
        raise RuntimeError('No saved run found')
    else:common.atomic_json(mp,config)
    results=common.load_results(out/'blink.jsonl')
    assert set(results).issubset({r['idx'] for r in data})
    def save():
        s=score(results,len(data));common.atomic_json(out/'blink.summary.json',s);return s
    save()
    if a.score_only or len(results)==len(data):
        print(json.dumps(save(),indent=2));return
    common.gpu_guard()
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import torch
    from PIL import Image
    from modelscope import AutoModelForCausalLM
    from transformers import set_seed
    set_seed(a.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    print('Loading Ovis2.5-9B; single pass, greedy, thinking_budget=1536 under ovis_compare_greedy_v1.',flush=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,
           torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    protocol_context = dict(stage='evaluate')
    install_ovis(model, out, lambda: dict(protocol_context))
    with (out/'blink.jsonl').open('a',buffering=1) as f:
        for i,row in enumerate(data):
            key=row['idx']
            if key in results:continue
            try:
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                with Image.open(io.BytesIO(base64.b64decode(row['image_b64']))) as raw_image:
                    image=raw_image.convert('RGB')
                prompt=row['prompt']+SUFFIX
                messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
                protocol_context.update(id=str(row['idx']))
                ids,pixels,grids=model.preprocess_inputs(messages=messages,add_generation_prompt=True,enable_thinking=True)
                ids=ids.cuda();pixels=pixels.cuda() if pixels is not None else None;grids=grids.cuda() if grids is not None else None
                with torch.inference_mode():
                    generated=model.generate(inputs=ids,pixel_values=pixels,grid_thws=grids,
                        enable_thinking=True,enable_thinking_budget=True,thinking_budget=a.thinking_budget,
                        max_new_tokens=a.max_new_tokens,do_sample=False)
                tokens=generated[0]
                if len(tokens)>=ids.shape[1] and torch.equal(tokens[:ids.shape[1]],ids[0]):tokens=tokens[ids.shape[1]:]
                response=model.text_tokenizer.decode(tokens,skip_special_tokens=True)
                prediction=parse_answer(response);torch.cuda.synchronize()
                r={k:v for k,v in row.items() if k!='image_b64'}
                r.update(key=key,input_prompt=prompt,response=response,prediction=prediction,
                    correct=prediction==row['answer'].strip('()'),input_tokens=ids.numel(),output_tokens=len(tokens),
                    elapsed_seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                    output_at_token_limit=len(tokens)>=a.max_new_tokens)
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());results[key]=r
                print(f'{i+1}/{len(data)} {key}: pred={prediction} correct={r["correct"]} seconds={r["elapsed_seconds"]:.1f}',flush=True)
                del ids,pixels,grids,generated,tokens,image,messages
                if len(results)%10==0:save()
            except Exception as exc:
                with (out/'errors.jsonl').open('a') as ef:
                    ef.write(json.dumps(dict(key=key,error=repr(exc),time=time.time()))+'\n')
                save();raise
    print(json.dumps(save(),indent=2),flush=True)


if __name__=='__main__':main()

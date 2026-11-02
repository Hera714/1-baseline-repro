#!/usr/bin/env python3
"""Single-pass Ovis2.5 evaluation; no GPU imports in --check-only/--score-only."""
import sys as _protocol_sys
from pathlib import Path as _ProtocolPath
_protocol_sys.path.insert(0,str(_ProtocolPath(__file__).resolve().parent.parent))
from protocol import install_ovis
import argparse
import base64
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/root/benchmark-data')
MANUAL_COT = """You are a spatial-reasoning assistant.

Task
----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).
Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”"""
FORMAT = "End your answer with a separate line formatted exactly as:\n\nAnswer: X\nwhere X ∈ {A, B, C, D}."
PATTERN = re.compile(r'Answer\s*:\s*([A-D])\b', re.IGNORECASE)
GROUPS = {
    'height': ['height_higher'],
    'location': ['location_above', 'location_closer_to_camera', 'location_next_to'],
    'orientation': ['orientation_in_front_of', 'orientation_on_the_left', 'orientation_viewpoint'],
    'multi_object': ['multi_object_closer_to', 'multi_object_facing', 'multi_object_viewpoint_towards_object', 'multi_object_parallel', 'multi_object_same_direction'],
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_path(name, args):
    return Path(args.omni_root) / 'data.json' if name == 'omnispatial' else Path(args.sr_tsv)


def samples(name, args):
    if name == 'omnispatial':
        for row in json.loads(source_path(name, args).read_text()):
            assert isinstance(row['answer'], int) and 0 <= row['answer'] <= 3
            assert len(row['options']) in (0, 2, 4)
            if row['options']:
                assert row['answer'] < len(row['options'])
            key = row['task_type'] + '/' + row['id']
            image = Path(args.omni_root) / row['task_type'] / (row['id'].split('_')[0] + '.png')
            assert image.is_file(), image
            prompt = MANUAL_COT + '\n' + FORMAT + '\n\n' + row['question']
            prompt += ''.join('\n' + chr(65+i) + '. ' + v for i, v in enumerate(row['options']))
            yield dict(key=key, qid=row['id'], group=key, category=row['task_type'],
                       subcategory=row['sub_task_type'], answer=chr(65+row['answer']),
                       prompt=prompt, image_path=str(image), option_count=len(row['options']))
    else:
        csv.field_size_limit(50 * 1024 * 1024)
        with source_path(name, args).open(newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                options = {k: row[k] for k in 'ABCD' if row.get(k, '').strip()}
                assert row['answer'] in options, row['qid']
                qid = row['qid']
                # Exact grouping rule from compute_3drbench_results_circular.py.
                group = qid[:-2] if len(qid) >= 2 and qid[-2] == '-' else qid
                prompt = 'Question: ' + row['question'] + '\nOptions:\n'
                prompt += ''.join(k + '. ' + v + '\n' for k, v in options.items())
                prompt += 'Please select the correct answer from the options above.\n' + FORMAT
                assert row['image'], qid
                yield dict(key=row['index'], qid=qid, group=group, category=row['category'],
                           subcategory=row['category'], answer=row['answer'], prompt=prompt,
                           image_b64=row['image'], option_count=len(options))


def scan(name, args):
    keys, groups, cats, opts = set(), collections.Counter(), collections.Counter(), collections.Counter()
    for s in samples(name, args):
        assert s['key'] not in keys, s['key']
        keys.add(s['key']); groups[s['group']] += 1
        cats[s['category']] += 1; opts[s['option_count']] += 1
    return dict(rows=len(keys), groups=len(groups), categories=dict(cats),
                option_counts=dict(opts), group_sizes=dict(collections.Counter(groups.values()))), groups


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load_results(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open('rb+') as f:
        while True:
            pos = f.tell(); line = f.readline()
            if not line:
                break
            try:
                r = json.loads(line)
            except (ValueError, UnicodeError):
                # Only a torn final write may be removed during resumption.
                if f.read():
                    raise RuntimeError('Corrupt results before final line: ' + str(path))
                f.truncate(pos)
                break
            if not line.endswith(b'\n'):
                f.seek(0, 2); f.write(b'\n')
            if r['key'] in rows:
                raise RuntimeError('Duplicate result: ' + r['key'])
            rows[r['key']] = r
    return rows


def summary(name, rows, expected, expected_groups):
    values = list(rows.values())
    def stats(v):
        return {'n': len(v), 'correct': sum(v), 'accuracy_percent': 100*sum(v)/len(v) if v else None}
    out = dict(benchmark=name, repeat=1, expected_rows=expected['rows'], processed_rows=len(values),
               complete=len(values) == expected['rows'],
               unparsed_rows=sum(not x['parsed'] for x in values),
               row_accuracy=stats([x['correct'] for x in values]))
    if name == 'omnispatial':
        out['scoring'] = 'official re parser including fallback A; single pass'
        out['overall'] = out['row_accuracy']
        out['categories'] = {k: stats([x['correct'] for x in values if x['category'] == k])
                             for k in sorted({x['category'] for x in values})}
        out['subcategories'] = {k: stats([x['correct'] for x in values if x['subcategory'] == k])
                                for k in sorted({x['subcategory'] for x in values})}
    else:
        grouped = collections.defaultdict(list)
        for r in values:
            grouped[r['group']].append(r)
        complete = {k: v for k, v in grouped.items() if len(v) == expected_groups[k]}
        out['scoring'] = 'local Answer:X exact match; official circular group aggregation; no LLM judge'
        out['complete_groups'] = len(complete)
        out['expected_groups'] = len(expected_groups)
        out['overall'] = stats([all(x['correct'] for x in v) for v in complete.values()])
        out['categories'] = {k: stats([all(x['correct'] for x in v) for v in complete.values()
                                      if v[0]['category'] in subtypes]) for k, subtypes in GROUPS.items()}
        out['subcategories'] = {k: stats([all(x['correct'] for x in v) for v in complete.values()
                                         if v[0]['category'] == k]) for k in sum(GROUPS.values(), [])}
    return out


def gpu_guard():
    # Do not compete with the existing router-generation process.
    running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if running:
        raise RuntimeError('GPU already has compute processes (PIDs: ' + running.replace('\n', ', ') + '). Wait for them to finish; no model loaded.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['omnispatial', '3dsrbench', 'both'], default='both')
    p.add_argument('--model', default='/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B')
    p.add_argument('--omni-root', default=str(ROOT/'OmniSpatial/dataset/OmniSpatial-test'))
    p.add_argument('--sr-tsv', default=str(ROOT/'3DSRBench/3dsrbench_v1_vlmevalkit_circular.tsv'))
    p.add_argument('--output', default='/root/autodl-fs/baselines/cost_bench/ovis_compare_greedy_v1/runs/omnispatial_native')
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--thinking-budget', type=int, default=1536)
    p.add_argument('--no-thinking', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=0, help='Diagnostic subset only; not a benchmark result')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--score-only', action='store_true')
    args = p.parse_args()
    if args.limit < 0 or (not args.no_thinking and args.max_new_tokens <= args.thinking_budget + 25):
        p.error('Invalid limit or token budget: max_new_tokens must exceed thinking_budget + 25')
    names = ['omnispatial', '3dsrbench'] if args.benchmark == 'both' else [args.benchmark]
    output = Path(args.output)
    expected = {name: scan(name, args) for name in names}
    for name in names:
        print(name, json.dumps(expected[name][0]), flush=True)
    if args.check_only:
        from PIL import Image
        for name in names:
            checked = 0
            for s in samples(name, args):
                if name == '3dsrbench' and checked >= 4:
                    break
                if name == 'omnispatial' and s['option_count'] != 0:
                    continue
                image = Image.open(io.BytesIO(base64.b64decode(s['image_b64'], validate=True))) if name == '3dsrbench' else Image.open(s['image_path'])
                image.verify(); checked += 1
            print(name, 'sample image checks OK:', checked, flush=True)
        print('CHECK PASSED; no model loaded, no GPU used.', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_results = {}
    for name in names:
        meta = dict(benchmark=name, repeat=1, model=str(Path(args.model).resolve()),
                    model_config_sha256=digest(Path(args.model)/'config.json'),
                    script_sha256=digest(__file__), data_sha256=digest(source_path(name, args)),
                    max_new_tokens=args.max_new_tokens, thinking_budget=args.thinking_budget,
                    enable_thinking=not args.no_thinking, do_sample=False, seed=args.seed, limit=args.limit,
                    dataset=expected[name][0])
        mp = output/(name+'.meta.json')
        if mp.exists():
            if json.loads(mp.read_text()) != meta:
                raise RuntimeError('Settings/data/code changed; use a NEW --output directory: ' + str(mp))
        elif args.score_only:
            raise RuntimeError('No metadata found: ' + str(mp))
        else:
            atomic_json(mp, meta)
        all_results[name] = load_results(output/(name+'.jsonl'))
        atomic_json(output/(name+'.summary.json'), summary(name, all_results[name], *expected[name]))
    if args.score_only:
        print('Scores refreshed without loading model.'); return
    if all(len(all_results[n]) >= (min(args.limit, expected[n][0]['rows']) if args.limit else expected[n][0]['rows']) for n in names):
        print('All requested samples already completed.'); return
    gpu_guard()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from PIL import Image
    from modelscope import AutoModelForCausalLM
    from transformers import set_seed
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU inference is disabled')
    print('Loading Ovis once for sequential evaluation...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
             local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    protocol_context = dict(stage='evaluate')
    install_ovis(model, output, lambda: dict(protocol_context))
    for name in names:
        results = all_results[name]
        with (output/(name+'.jsonl')).open('a', buffering=1) as fout:
            for i, sample in enumerate(samples(name, args)):
                if args.limit and i >= args.limit:
                    break
                if sample['key'] in results:
                    continue
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
                try:
                    if name == '3dsrbench':
                        image = Image.open(io.BytesIO(base64.b64decode(sample['image_b64'], validate=True))).convert('RGB')
                    else:
                        with Image.open(sample['image_path']) as original:
                            image = original.convert('RGB')
                    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': sample['prompt']}]}]
                    protocol_context.update(id=str(sample['key']))
                    ids, pixels, grids = model.preprocess_inputs(messages=messages, add_generation_prompt=True, enable_thinking=not args.no_thinking)
                    ids = ids.cuda(); pixels = pixels.cuda() if pixels is not None else None
                    grids = grids.cuda() if grids is not None else None
                    with torch.inference_mode():
                        generated = model.generate(inputs=ids, pixel_values=pixels, grid_thws=grids,
                            enable_thinking=not args.no_thinking, enable_thinking_budget=not args.no_thinking,
                            thinking_budget=args.thinking_budget, max_new_tokens=args.max_new_tokens, do_sample=False)
                    tokens = generated[0]
                    if len(tokens) >= ids.shape[1] and torch.equal(tokens[:ids.shape[1]], ids[0]):
                        tokens = tokens[ids.shape[1]:]
                    response = model.text_tokenizer.decode(tokens, skip_special_tokens=True)
                    # Match the official OmniSpatial parser over the full response.
                    found = PATTERN.findall(response)
                    pred = found[-1] if found else ('A' if name == 'omnispatial' else None)
                    torch.cuda.synchronize()
                    record = {k: v for k, v in sample.items() if k not in ('image_b64',)}
                    record.update(response=response, prediction=pred, parsed=bool(found), correct=pred == sample['answer'],
                                  input_tokens=int(ids.numel()), output_tokens=len(tokens),
                                  elapsed_seconds=time.perf_counter()-start,
                                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                                  output_at_token_limit=len(tokens) >= args.max_new_tokens)
                    fout.write(json.dumps(record, ensure_ascii=False)+'\n'); fout.flush(); os.fsync(fout.fileno())
                    results[sample['key']] = record
                    print(f'{name} {i+1}/{expected[name][0]["rows"]} key={sample["key"]} pred={pred} hit={record["correct"]} seconds={record["elapsed_seconds"]:.1f}', flush=True)
                    del ids, pixels, grids, generated, tokens, messages, image
                    if len(results) % 25 == 0:
                        atomic_json(output/(name+'.summary.json'), summary(name, results, *expected[name]))
                except Exception as exc:
                    with (output/'errors.jsonl').open('a') as ef:
                        ef.write(json.dumps({'benchmark': name, 'key': sample['key'], 'error': repr(exc), 'time': time.time()})+'\n')
                    atomic_json(output/(name+'.summary.json'), summary(name, results, *expected[name]))
                    raise  # Never skip failed samples; rerun the same command to resume.
        score = summary(name, results, *expected[name])
        atomic_json(output/(name+'.summary.json'), score)
        print(json.dumps(score, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

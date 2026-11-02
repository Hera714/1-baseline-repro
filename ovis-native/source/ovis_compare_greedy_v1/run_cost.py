#!/usr/bin/env python3
"""Profile frozen adapters in memory, without editing original experiments."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import sys
from protocol import parse_item, format_options
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-fs/baselines')
DEFAULT_DATA = ROOT / 'spatialpin/data/spatiallab_mcq.json'


def manifest(args):
    rows = json.loads(args.data.read_text(encoding='utf-8'))
    rng = random.Random(20261020)
    categories = defaultdict(lambda: defaultdict(list))
    for row in rows:
        categories[row['Category']][row.get('Sub_Category', '')].append(row)
    chosen = []
    for category in sorted(categories):
        groups = categories[category]
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < args.per_category:
            progress = False
            for key in sorted(groups):
                if groups[key] and len(selected) < args.per_category:
                    selected.append(groups[key].pop())
                    progress = True
            if not progress:
                break
        chosen.extend(selected)
    warmup = [dict(chosen[i], id='warmup_' + str(chosen[i]['id'])) for i in range(min(2, len(chosen)))]
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'samples.json'
    content = json.dumps(warmup + chosen, ensure_ascii=False, indent=2) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != content:
        raise RuntimeError('Existing fixed manifest differs; choose a new directory')
    target.write_text(content, encoding='utf-8')
    (args.output / 'manifest.json').write_text(json.dumps(dict(seed=20261020,
        selection='category-balanced, subcategory round-robin, independent of predictions and timing',
        measured_ids=[str(x['id']) for x in chosen], warmup_ids=[str(x['id']) for x in warmup],
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        measured_samples=len(chosen), batch_size=1), indent=2), encoding='utf-8')
    print(target, 'measured=', len(chosen), 'warmup=', len(warmup))


class Instrument(ast.NodeTransformer):
    LOADS = {'from_pretrained', 'load_pretrained_model', 'get_depth_predictor',
             'load_spatialpin_models', 'Ovis', 'DepthKosmosCaptioner'}
    COMPONENTS = {'run_detection', 'get_depth_map', 'estimate_inverse_depth', 'make_region_masks',
                  'process_images', 'tokenizer_image_token', 'build_evidence_panel', 'save_artifacts'}

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
        if name in self.LOADS or name in self.COMPONENTS:
            wrapper = 'load' if name in self.LOADS else 'call'
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='_profile', ctx=ast.Load()),
                attr=wrapper, ctx=ast.Load()), args=[ast.Constant(name), node.func] + node.args,
                keywords=node.keywords), node)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == 'started' for t in node.targets):
            # All frozen adapters name the current string ID item_id.
            return [ast.parse('_profile.begin(item_id)').body[0], node]
        def elapsed_target(t):
            if not (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'record'):
                return False
            part = t.slice.value if isinstance(t.slice, ast.Index) else t.slice
            return isinstance(part, ast.Constant) and part.value == 'elapsed_seconds'
        if any(elapsed_target(t) for t in node.targets):
            return [node, ast.parse('_profile.end(record)').body[0]]
        return node


def profile_adapter(args):
    from profile_runtime import Recorder
    if args.method == 'spatialrgpt' and args.stage == 'evaluate':
        from offline_depth import install
        install()
    sources = Path(__file__).parent / 'sources'
    source = sources / ('spatialpin.py' if args.method == 'spatialpin' else
                        'prepare_regions.py' if args.stage == 'regions' else 'spatialrgpt.py')
    profiler = Recorder(args.output / (args.method + '_' + args.stage + '.cost.jsonl'), args.method, args.stage)
    namespace = dict(__name__='cost_adapter', __file__=str(source), _profile=profiler)
    tree = ast.fix_missing_locations(Instrument().visit(ast.parse(source.read_text(encoding='utf-8'))))
    try:
        exec(compile(tree, str(source), 'exec'), namespace)
        sys.argv = [str(source)] + args.adapter_args
        if args.method == 'spatialpin':
            sys.argv.insert(1, args.stage)
        namespace['main']()
    finally:
        profiler.finish(source)


def native(args):
    import re
    import torch
    from PIL import Image
    from modelscope import AutoModelForCausalLM
    from profile_runtime import Recorder
    from online_cost import DeviceMemory
    profiler = Recorder(args.output / 'ovis_native_evaluate.cost.jsonl', 'ovis_native', 'evaluate')
    memory = DeviceMemory()
    try:
        model = profiler.load('Ovis', lambda: AutoModelForCausalLM.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation='flash_attention_2').cuda().eval())
        profiler.sync()
        rows = json.loads(args.data.read_text(encoding='utf-8'))
        with (args.output / 'ovis_native_predictions.jsonl').open('x', encoding='utf-8') as out:
            for item in rows:
                profiler.begin(item['id'])
                memory.start()
                record = dict(id=str(item['id']), status='ok')
                try:
                    image = Image.open(args.image_dir / item['Image_Filename']).convert('RGB')
                    options = format_options(item)
                    prompt = ('Answer the multiple-choice question using the image. Provide step-by-step reasoning '
                              'in a <think>...</think> block, then give exactly one choice as '
                              '<answer>N</answer>, where N is 1, 2, 3, or 4.\n\nQuestion:\n' +
                              item['Question'] + '\nOptions:\n' + options + '\nFinal answer:')
                    messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=prompt)])]
                    ids, pixels, grid = model.preprocess_inputs(messages=messages,
                        add_generation_prompt=True, enable_thinking=True)
                    with torch.no_grad():
                        output = model.generate(inputs=ids.cuda(), pixel_values=pixels.cuda() if pixels is not None else None,
                        grid_thws=grid.cuda() if grid is not None else None, max_new_tokens=args.max_new_tokens,
                        enable_thinking=True, enable_thinking_budget=True,
                        thinking_budget=max(256, int(args.max_new_tokens * .75)))
                    raw = model.text_tokenizer.decode(output[0], skip_special_tokens=True)
                    prediction, policy = parse_item(raw, item)
                    record.update(prompt=prompt, raw_response=raw, prediction=prediction, parse_method=policy)
                    record['is_correct'] = record['prediction'] == str(item.get('Answer'))
                except Exception as exc:
                    record.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
                profiler.current['peak_device_used_bytes_nvml_sampled_20ms'] = memory.stop()
                profiler.end(record)
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                print('Ovis sample', item['id'], record['status'], flush=True)
    finally:
        profiler.finish(Path(__file__))


def bydeway(args):
    from profile_runtime import Recorder
    from online_cost import DeviceMemory
    # The deployed copy differs only in environment paths and cwd handling.
    project = ROOT / 'bydeway/ByDeWay-main'
    sys.path[:0] = [str(project), str(project / 'Depth-Anything-V2')]
    source = Path(__file__).parent / 'bydeway_adapter.py'
    profiler = Recorder(args.output / 'bydeway_evaluate.cost.jsonl', 'bydeway', 'evaluate')
    memory = DeviceMemory()
    try:
        spec = importlib.util.spec_from_file_location('bydeway_benchmark', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = profiler.load('ByDeWayOvisRunner', module.ByDeWayOvisRunner, model_path=str(args.model))
        # Recursive attachment records Kosmos and Ovis as separate generation models.
        profiler.attach(runner.depth_captioner, 'ByDeWay.depth_captioner')
        profiler.attach(runner.ovis_model, 'ByDeWay.Ovis')
        from PIL import Image
        with (args.output / 'bydeway_predictions.jsonl').open('x', encoding='utf-8') as out:
            for item in json.loads(args.data.read_text(encoding='utf-8')):
                profiler.begin(item['id'])
                memory.start()
                record = dict(id=str(item['id']), status='ok')
                try:
                    image = Image.open(args.image_dir / item['Image_Filename']).convert('RGB')
                    options = format_options(item)
                    record.update(runner.infer(image, item['Question'], options, spatial_relations_text=''))
                    record['prediction'], record['parse_method'] = parse_item(record['raw_response'], item)
                    record['is_correct'] = record['prediction'] == str(item.get('Answer'))
                except Exception as exc:
                    record.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
                profiler.current['peak_device_used_bytes_nvml_sampled_20ms'] = memory.stop()
                profiler.end(record)
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                print('ByDeWay sample', item['id'], record['status'], flush=True)
    finally:
        profiler.finish(source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['manifest', 'spatialpin', 'spatialrgpt', 'ovis_native', 'bydeway'], required=True)
    p.add_argument('--stage', default='evaluate')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--image-dir', type=Path, default=Path('/root/autodl-fs/img'))
    p.add_argument('--model', type=Path, default=Path('/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B'))
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--per-category', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('adapter_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.adapter_args[:1] == ['--']:
        args.adapter_args = args.adapter_args[1:]
    if args.method == 'manifest':
        manifest(args)
    elif args.method == 'ovis_native':
        native(args)
    elif args.method == 'bydeway':
        bydeway(args)
    else:
        profile_adapter(args)


if __name__ == '__main__':
    main()

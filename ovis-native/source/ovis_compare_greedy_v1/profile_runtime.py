"""Synchronized GPU profiling; inclusive component timings are not additive."""
import contextlib
import functools
import hashlib
import json
import os
import platform
import socket
import time
from protocol import reset_seed, install_ovis
from pathlib import Path


class Recorder:
    def __init__(self, output, method, stage):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable: start this instance with its GPU before profiling')
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('x', encoding='utf-8')
        self.method, self.stage = method, stage
        self.current = None
        self.load_events = []
        self.stack = []
        self.hooked = set()
        self.models = []
        self.cache_loads = False
        self.load_cache = {}
        self.start = time.perf_counter()
        self.sync()
        torch.cuda.reset_peak_memory_stats()
        gpu = torch.cuda.get_device_properties(0)
        self.meta = dict(method=method, stage=stage, batch_size=1,
                         host=socket.gethostname(), gpu=gpu.name,
                         gpu_total_bytes=gpu.total_memory, torch=torch.__version__,
                         compiled_cuda=torch.version.cuda, python=platform.python_version(),
                         torch_num_threads=torch.get_num_threads(),
                         synchronized_component_profiling=True,
                         environment={k: os.environ.get(k) for k in
                                      ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                       'OPENBLAS_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']})

    def sync(self):
        self.torch.cuda.synchronize()

    def memory(self):
        t = self.torch.cuda
        return dict(peak_allocated_bytes=t.max_memory_allocated(),
                    peak_reserved_bytes=t.max_memory_reserved(),
                    allocated_bytes=t.memory_allocated(), reserved_bytes=t.memory_reserved())

    def begin(self, item_id):
        reset_seed()
        if self.current is not None:
            raise RuntimeError('Previous sample was not finalized')
        self.sync()
        if not hasattr(self, 'startup'):
            self.startup = dict(seconds_until_first_sample=time.perf_counter() - self.start,
                                memory=self.memory())
        self.torch.cuda.reset_peak_memory_stats()
        self.current = dict(id=str(item_id), warmup=str(item_id).startswith('warmup_'),
                            events=[], started=time.perf_counter())

    def end(self, record):
        self.sync()
        row = self.current
        row['sample_seconds'] = time.perf_counter() - row.pop('started')
        row.update(self.memory())
        row.update(method=self.method, stage=self.stage, status=record.get('status', 'ok'),
                   prediction=record.get('prediction', record.get('pred_answer')),
                   parse_method=record.get('parse_method'), error=record.get('error'))
        self.handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.handle.flush()
        self.current = None

    @contextlib.contextmanager
    def event(self, name, kind='component'):
        self.sync()
        target = self.current['events'] if self.current else self.load_events
        event = dict(name=name, kind=kind, parent=self.stack[-1]['name'] if self.stack else None)
        target.append(event)
        self.stack.append(event)
        start = time.perf_counter()
        try:
            yield event
            event['status'] = 'ok'
        except BaseException as exc:
            event.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            self.sync()
            event['seconds'] = time.perf_counter() - start
            self.stack.pop()

    def call(self, name, fn, *args, **kwargs):
        with self.event(name):
            value = fn(*args, **kwargs)
        return value

    def load(self, name, fn, *args, **kwargs):
        if name == 'load_spatialpin_models' and args:
            # The caller passes the whole argparse Namespace. Its per-request
            # slice/output fields change, but the loaded models do not.
            config = args[0]
            identity = (str(getattr(config, 'device', 'cuda')),
                        str(Path(config.depth_checkpoint).resolve()))
        else:
            identity = repr(args)
        key = (name, repr(fn), identity, repr(sorted(kwargs.items())))
        if self.cache_loads and key in self.load_cache:
            return self.load_cache[key]
        with self.event(name, 'load'):
            value = fn(*args, **kwargs)
        self.attach(value, name)
        if self.cache_loads:
            self.load_cache[key] = value
        return value

    def attach(self, value, name, depth=0):
        if value is None or depth > 4 or id(value) in self.hooked:
            return
        self.hooked.add(id(value))
        if type(value).__name__ == 'Ovis2_5':
            install_ovis(value, self.path.parent, lambda: dict(id=(self.current or {}).get('id'), stage=self.stage))
        if isinstance(value, (list, tuple)):
            tokenizer = value[0] if value and hasattr(value[0], 'decode') else None
            for i, obj in enumerate(value):
                if tokenizer is not None and hasattr(obj, 'generate'):
                    obj._cost_tokenizer = tokenizer
                self.attach(obj, name + '.' + str(i), depth + 1)
            return
        if isinstance(value, self.torch.nn.Module):
            config = getattr(value, 'config', None)
            info = dict(label=name, class_name=type(value).__name__,
                        model_path=getattr(config, '_name_or_path', None))
            if hasattr(value, 'parameters'):
                info['parameters'] = sum(p.numel() for p in value.parameters())
            self.models.append(info)
        if hasattr(value, 'generate') and callable(value.generate):
            original = value.generate
            @functools.wraps(original)
            def generate(*args, _generate=original, **kwargs):
                kind = 'llm_generate' if name.endswith('.llm') else 'generate'
                with self.event(name + '.generate', kind) as ev:
                    ids = kwargs.get('inputs')
                    if ids is None:
                        ids = kwargs.get('input_ids')
                    if ids is None and args and self.torch.is_tensor(args[0]):
                        ids = args[0]
                    embeds = kwargs.get('inputs_embeds')
                    if ids is not None:
                        ev['input_sequence_tokens'] = int(ids.numel())
                        ev['nonnegative_input_tokens'] = int((ids >= 0).sum().item())
                        negative, counts = self.torch.unique(ids[ids < 0], return_counts=True)
                        ev['negative_visual_placeholders'] = {str(k): v for k, v in zip(negative.tolist(), counts.tolist())}
                        if '-300' in ev['negative_visual_placeholders']:
                            ev['ovis_image_atom_tokens'] = ev['negative_visual_placeholders']['-300']
                            ev['ovis_visual_indicator_tokens'] = sum(v for k, v in ev['negative_visual_placeholders'].items() if k != '-300')
                        ev['input_token_ids'] = ids.detach().cpu().tolist()
                    if embeds is not None:
                        ev['effective_input_embedding_tokens'] = int(embeds.shape[-2])
                    image_mask = kwargs.get('image_embeds_position_mask')
                    if image_mask is not None:
                        ev['kosmos_image_embedding_positions'] = int(image_mask.sum().item())
                        if 'nonnegative_input_tokens' in ev:
                            ev['input_text_and_control_tokens'] = ev['nonnegative_input_tokens'] - ev['kosmos_image_embedding_positions']
                    elif 'nonnegative_input_tokens' in ev:
                        ev['input_text_and_control_tokens'] = ev['nonnegative_input_tokens']
                    for field in ['pixel_values', 'images', 'depths', 'grid_thws']:
                        tensor = kwargs.get(field)
                        if self.torch.is_tensor(tensor):
                            ev[field + '_shape'] = list(tensor.shape)
                    ev['generation_kwargs'] = {k: kwargs.get(k) for k in
                        ['max_new_tokens', 'do_sample', 'temperature', 'num_beams',
                         'enable_thinking', 'enable_thinking_budget', 'thinking_budget']}
                    cfg = getattr(value, 'generation_config', None)
                    if cfg is not None:
                        ev['model_generation_defaults'] = {k: getattr(cfg, k, None) for k in
                                                           ['do_sample', 'temperature', 'top_p', 'top_k', 'num_beams']}
                    result = _generate(*args, **kwargs)
                    sequences = result.sequences if hasattr(result, 'sequences') else result
                    if self.torch.is_tensor(sequences):
                        out = sequences[0]
                        if ids is not None and out.numel() >= ids.shape[-1] and self.torch.equal(out[:ids.shape[-1]], ids[0]):
                            out = out[ids.shape[-1]:]
                        ev['returned_output_tokens'] = int(out.numel())
                        ev['output_token_ids'] = out.detach().cpu().tolist()
                    return result
            value.generate = generate
        for attr in ['preprocess_inputs', 'prepare_inputs_labels_for_multimodal', 'merge_multimodal',
                     'infer_image', 'predict', 'predict_dino', 'predict_sam', 'set_image', 'predict_torch', 'get_caption_with_depth']:
            original = getattr(value, attr, None)
            if not callable(original):
                continue
            def make_wrapper(fn, method):
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    with self.event(name + '.' + method) as ev:
                        result = fn(*args, **kwargs)
                        if method == 'preprocess_inputs' and isinstance(result, tuple):
                            ev['input_shape'] = list(result[0].shape)
                            if len(result) > 2 and result[2] is not None:
                                ev['image_grid_thws'] = result[2].detach().cpu().tolist()
                        if method == 'merge_multimodal' and self.torch.is_tensor(result):
                            ev['effective_input_embedding_tokens'] = int(result.shape[-2])
                        if method == 'prepare_inputs_labels_for_multimodal' and isinstance(result, tuple) and len(result) > 4 and result[4] is not None:
                            ev['effective_input_embedding_tokens'] = int(result[4].shape[-2])
                            original_ids = kwargs.get('input_ids')
                            if original_ids is None and args:
                                original_ids = args[0]
                            if original_ids is not None:
                                ev['nonimage_input_positions'] = int((original_ids >= 0).sum().item())
                                ev['visual_expansion_tokens_after_truncation'] = int(result[4].shape[-2]) - ev['nonimage_input_positions']
                                ev['visual_expansion_note'] = 'Difference after model truncation; inspect effective sequence and configured context limit'
                            if hasattr(value, 'get_vision_tower') and original_ids is not None:
                                cfg = value.get_vision_tower().config
                                ev['regional_mask_positions'] = int((original_ids == cfg.llm_mask_token_id).sum().item())
                                ev['regional_depth_positions'] = int((original_ids == cfg.llm_depth_token_id).sum().item())
                        return result
                return wrapped
            setattr(value, attr, make_wrapper(original, attr))
        for attr in ['model', 'llm', 'groundingdino', 'sam', 'sam_predictor', 'predictor', 'depth_context', 'captioner']:
            obj = getattr(value, attr, None)
            if obj is not value:
                self.attach(obj, name + '.' + attr, depth + 1)

    def finish(self, source):
        self.sync()
        self.meta.update(process_profile_seconds=time.perf_counter() - self.start,
                         startup=getattr(self, 'startup', None), load_events=self.load_events,
                         models=self.models, argv=__import__('sys').argv,
                         source_file=str(source), source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
        here = Path(__file__).parent
        self.meta['profiler_source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in here.glob('*.py')}
        self.meta['related_source_hashes'] = {}
        for name in ['kar3d.perception', 'llava.model.builder', 'llava.model.llava_arch']:
            module = __import__('sys').modules.get(name)
            path = Path(module.__file__) if module is not None and getattr(module, '__file__', None) else None
            if path is not None and path.is_file():
                self.meta['related_source_hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.path.with_suffix('.meta.json').write_text(json.dumps(self.meta, indent=2), encoding='utf-8')
        self.handle.close()

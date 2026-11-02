#!/usr/bin/env python3
"""Single-GPU online requests through resident workers in original environments."""
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

BASE = Path('/root/autodl-fs/baselines')
BENCH = Path(__file__).resolve().parent
DATA = BENCH / 'manifest60/samples.json'
SPIN = BASE / 'spatialpin/SpatialPIN/spatialpin'
SRGPT = BASE / 'spatialrgpt/SpatialRGPT'
MODEL = Path('/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B')
KAR = BENCH / 'frozen_kar'
PYTHON = dict(ovis='/root/ovis-compare-greedy-v1-env/bin/python',
              pin=str(BASE / 'envs/spatialpin/bin/python'),
              regions='/root/cost-detector-env/bin/python',
              rgpt=str(BASE / 'envs/spatialrgpt/bin/python'))


class DeviceMemory:
    """NVML polling measures total device memory, including all resident workers."""
    def __init__(self):
        self.library = None
        self.peak = None
        try:
            lib = ctypes.CDLL('libnvidia-ml.so.1')
            if lib.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return
            self.library, self.handle = lib, handle
        except OSError:
            pass

    def read(self):
        if self.library is None:
            return None
        class Memory(ctypes.Structure):
            _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
        info = Memory()
        if self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) == 0:
            return int(info.used)

    def start(self):
        self.running = True
        self.peak = self.read()
        def poll():
            while self.running:
                value = self.read()
                if value is not None:
                    self.peak = max(self.peak or 0, value)
                time.sleep(.02)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        value = self.read()
        if value is not None:
            self.peak = max(self.peak or 0, value)
        return self.peak


def worker(args):
    from profile_runtime import Recorder
    from run_cost import Instrument
    method = 'spatialpin' if args.worker in ['ovis', 'pin'] else 'spatialrgpt'
    if args.worker == 'rgpt':
        from offline_depth import install
        install()
    source = BENCH / 'sources' / dict(ovis='spatialpin.py', pin='spatialpin.py',
                                    regions='prepare_regions.py', rgpt='spatialrgpt.py')[args.worker]
    recorder = Recorder(args.output / (method + '_' + args.worker + '.cost.jsonl'), method, args.worker)
    recorder.cache_loads = True
    namespace = dict(__name__='online_adapter', __file__=str(source), _profile=recorder)
    tree = ast.fix_missing_locations(Instrument().visit(ast.parse(source.read_text(encoding='utf-8'))))
    exec(compile(tree, str(source), 'exec'), namespace)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('stop'):
                break
            recorder.stage = request['stage']
            sys.argv = [str(source)] + request['argv']
            try:
                namespace['main']()
                response = dict(ok=True)
            except BaseException as exc:
                response = dict(ok=False, error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
            print('COST_RPC ' + json.dumps(response), flush=True)
    finally:
        recorder.finish(source)


def cli(method, stage, index, output, data=DATA):
    shared = ['--data', str(data), '--image-dir', '/root/autodl-fs/img',
              '--start-index', str(index), '--end-index', str(index + 1)]
    if method == 'spatialpin':
        shared.insert(0, stage)
        if stage == 'discover':
            return shared + ['--model-path', str(MODEL), '--kar3d-root', str(KAR),
                             '--output', str(output / 'discoveries.jsonl')]
        if stage == 'prepare':
            return shared + ['--discoveries', str(output / 'discoveries.jsonl'),
                             '--depth-checkpoint', str(BASE / 'models/Depth-Anything/checkpoints/depth_anything_vitl14.pth'),
                             '--output', str(output / 'prepared.jsonl')]
        return shared + ['--model-path', str(MODEL), '--kar3d-root', str(KAR),
                         '--prepared', str(output / 'prepared.jsonl'), '--output-dir', str(output / 'spatialpin')]
    if stage == 'regions':
        return shared + ['--model', str(BASE / 'models/grounding-dino-tiny'),
                         '--output', str(output / 'regions.jsonl')]
    return shared + ['--regions', str(output / 'regions.jsonl'),
                     '--model-path', str(BASE / 'models/SpatialRGPT-VILA1.5-8B'),
                     '--depth-anything-path', str(BASE / 'models/Depth-Anything'),
                     '--save-artifacts', '--output-dir', str(output / 'spatialrgpt')]


def parent(args):
    from summarize import stats
    # Check before opening the coordinator file or overwriting any worker log.
    # A cancelled startup can leave empty cost files even without a prediction.
    if args.method == 'spatialpin':
        owned = ['spatialpin_online.jsonl', 'spatialpin_ovis.cost.jsonl', 'spatialpin_pin.cost.jsonl',
                 'discoveries.jsonl', 'prepared.jsonl', 'spatialpin/predictions.jsonl']
    else:
        owned = ['spatialrgpt_online.jsonl', 'spatialrgpt_regions.cost.jsonl', 'spatialrgpt_rgpt.cost.jsonl',
                 'regions.jsonl', 'spatialrgpt/predictions.jsonl']
    existing = [str(args.output / name) for name in owned if (args.output / name).exists()]
    if existing:
        raise RuntimeError('This method already has output files in COST_RUN, possibly from an interrupted startup. '
                           'Existing files and logs are preserved. Set COST_RUN to a fresh directory, for example '
                           '/root/autodl-fs/baselines/cost_bench/runs/profile60_v2. Files: ' + ', '.join(existing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: start this instance with its GPU before profiling')
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / (args.method + '_online.jsonl')
    stream = marker.open('x', encoding='utf-8')
    kinds = ['ovis', 'pin'] if args.method == 'spatialpin' else ['regions', 'rgpt']
    tasks = [('ovis', 'discover'), ('pin', 'prepare'), ('ovis', 'evaluate')] if args.method == 'spatialpin' else [('regions', 'regions'), ('rgpt', 'evaluate')]
    processes = {}
    logs = {}
    memory = DeviceMemory()
    start = time.perf_counter()
    timings = []
    peaks = []
    statuses = Counter()
    parse_failures = 0
    try:
        for kind in kinds:
            if not Path(PYTHON[kind]).is_file():
                raise FileNotFoundError(PYTHON[kind])
            env = dict(os.environ)
            env.pop('LD_PRELOAD', None)
            if kind == 'pin':
                env['PYTHONPATH'] = ':'.join(map(str, [SPIN, SPIN / 'main', SPIN / 'export']))
                env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
            else:
                env['PYTHONPATH'] = str(SRGPT if kind in ['regions', 'rgpt'] else KAR / 'src')
            logs[kind] = (args.output / (args.method + '_' + kind + '.worker.log')).open('w', encoding='utf-8')
            processes[kind] = subprocess.Popen([PYTHON[kind], '-u', str(Path(__file__)),
                '--worker', kind, '--output', str(args.output)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=logs[kind], text=True, bufsize=1, env=env)
        items = json.loads(args.data.read_text(encoding='utf-8'))
        if args.limit is not None:
            items = items[:args.limit]
        for index, item in enumerate(items):
            memory.start()
            begin = time.perf_counter()
            record = dict(id=str(item['id']), method=args.method, batch_size=1,
                          warmup=str(item['id']).startswith('warmup_'), status='ok', stages=[])
            fatal = False
            for kind, stage in tasks:
                proc = processes[kind]
                print('Sample ' + str(index + 1) + '/' + str(len(items)) + ' ' + str(item['id'])
                      + ': starting ' + stage + (' (initial model loading/warm-up)' if index < 2 else ''), flush=True)
                request = dict(stage=stage, argv=cli(args.method, stage, index, args.output, args.data))
                stage_start = time.perf_counter()
                proc.stdin.write(json.dumps(request) + '\n')
                proc.stdin.flush()
                response = None
                for line in proc.stdout:
                    if line.startswith('COST_RPC '):
                        response = json.loads(line[len('COST_RPC '):])
                        break
                    logs[kind].write(line)
                    logs[kind].flush()
                if response is None:
                    logs[kind].flush()
                    path = args.output / (args.method + '_' + kind + '.worker.log')
                    tail = path.read_text(encoding='utf-8', errors='replace')[-3000:]
                    raise RuntimeError('Worker exited: ' + kind + '; log: ' + str(path) + '\n' + tail)
                stage_files = dict(discover='discoveries.jsonl', prepare='prepared.jsonl',
                    regions='regions.jsonl', evaluate=args.method + '/predictions.jsonl')
                stage_file = args.output / stage_files[stage]
                if response['ok'] and stage_file.exists():
                    stage_rows = [json.loads(x) for x in stage_file.read_text(encoding='utf-8').splitlines() if x.strip()]
                    matches = [r for r in stage_rows if str(r['id']) == str(item['id'])]
                    if matches and matches[-1].get('status') != 'ok':
                        response.update(sample_status='error', sample_error=matches[-1].get('error'))
                record['stages'].append(dict(stage=stage, request_seconds=time.perf_counter() - stage_start, **response))
                if not response['ok']:
                    record.update(status='error', error=response.get('error'))
                    fatal = True
                    break
                if response.get('sample_status') == 'error':
                    record.update(status='error', error=response.get('sample_error'))
                    break
            record['online_wall_seconds'] = time.perf_counter() - begin
            record['peak_device_used_bytes_nvml_sampled_20ms'] = memory.stop()
            # Check adapter status too: original adapters catch inference exceptions internally.
            final_path = args.output / args.method / 'predictions.jsonl'
            if final_path.exists():
                matching = [json.loads(x) for x in final_path.read_text(encoding='utf-8').splitlines()
                            if x.strip() and str(json.loads(x)['id']) == str(item['id'])]
                if matching and matching[-1].get('status') != 'ok':
                    record.update(status='error', error=matching[-1].get('error'))
                elif not matching:
                    record['status'] = 'error'
                    record.setdefault('error', 'No final prediction for current ID')
                if matching:
                    final = matching[-1]
                    record['prediction'] = final.get('prediction', final.get('pred_answer'))
                    record['parse_method'] = final.get('parse_method')
                    record['parse_success'] = record['prediction'] not in [None, 'Error', 'Unknown']
            elif record['status'] == 'ok':
                record.update(status='error', error='No final predictions file')
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if not record['warmup']:
                timings.append(record['online_wall_seconds'])
                statuses[record['status']] += 1
                parse_failures += int(record.get('parse_success') is False)
                if record['peak_device_used_bytes_nvml_sampled_20ms'] is not None:
                    peaks.append(record['peak_device_used_bytes_nvml_sampled_20ms'])
            print(str(index + 1) + '/' + str(len(items)) + ' ' + record['id'] + ' ' + record['status'] + ' ' + str(round(record['online_wall_seconds'], 3)), flush=True)
            if fatal:
                raise RuntimeError('Worker initialization/entry failure; inspect worker.log: ' + str(record.get('error')))
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                try:
                    proc.stdin.write('{"stop": true}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    proc.terminate()
        for log in logs.values():
            log.close()
        stream.close()
        report = dict(method=args.method, measured_wall=stats(timings),
                      data_file=str(args.data), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      status_counts=dict(statuses), parse_failures=parse_failures,
                      peak_device_used_bytes_nvml_sampled_20ms=max(peaks) if peaks else None,
                      entire_session_seconds=time.perf_counter() - start,
                      note='Resident original-environment workers on one GPU; includes IPC, adapter reentry and file IO. First two requests warm up and load models; excluded from measured statistics. Device memory includes every GPU process; run alone.')
        marker.with_suffix('.summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['spatialpin', 'spatialrgpt'])
    parser.add_argument('--worker', choices=list(PYTHON))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DATA, help='Dataset with separately marked warm-up copies')
    parser.add_argument('--limit', type=int, help='Optional request count for smoke verification; default runs the complete fixed manifest')
    options = parser.parse_args()
    worker(options) if options.worker else parent(options)

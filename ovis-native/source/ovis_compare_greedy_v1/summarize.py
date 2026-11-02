#!/usr/bin/env python3
"""Summarize cost measurements; warm-ups excluded and failed samples retained."""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), p95=values[max(0, math.ceil(.95 * len(values)) - 1)],
                min=values[0], max=values[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = {'timing_units': 'seconds', 'p95': 'nearest rank', 'warmups_excluded': True,
              'failed_samples_included': True, 'component_times': 'inclusive; do not sum nested events',
              'memory_units': 'bytes; torch allocated/reserved, not total device memory', 'stages': {}}
    invalid = {'spatialpin'} if (args.directory / 'INVALID_SPATIALPIN_COST_RUN.md').exists() else set()
    result['invalid_methods_for_reporting'] = sorted(invalid)
    stages = defaultdict(dict)
    datasets = defaultdict(list)
    for file in sorted(args.directory.glob('*.cost.jsonl')):
        rows = [json.loads(x) for x in file.read_text(encoding='utf-8').splitlines() if x.strip()]
        rows = [r for r in rows if not r.get('warmup')]
        for row in rows:
            datasets[row['method'] + '_' + row['stage']].append(row)
    for label, rows in sorted(datasets.items()):
        events = [e for r in rows for e in r['events']]
        names = sorted({e['name'] for e in events})
        result['stages'][label] = dict(
            valid_for_reporting=rows[0]['method'] not in invalid,
            samples=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
            unparsed=sum(r.get('stage') == 'evaluate' and r.get('status') == 'ok'
                         and r.get('prediction') in [None, 'Unknown'] for r in rows),
            wall=stats([r['sample_seconds'] for r in rows]),
            peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in rows), default=None),
            peak_reserved_bytes=max((r['peak_reserved_bytes'] for r in rows), default=None),
            peak_device_used_bytes_nvml_sampled_20ms=max((r['peak_device_used_bytes_nvml_sampled_20ms'] for r in rows
                if r.get('peak_device_used_bytes_nvml_sampled_20ms') is not None), default=None),
            components={name: dict(calls=sum(e['name'] == name for e in events),
                failures=sum(e['name'] == name and e['status'] != 'ok' for e in events),
                time=stats([e['seconds'] for e in events if e['name'] == name]),
                returned_output_tokens=sum(e.get('returned_output_tokens', 0) for e in events if e['name'] == name),
                nonnegative_input_tokens=sum(e.get('nonnegative_input_tokens', 0) for e in events if e['name'] == name),
                input_text_and_control_tokens=sum(e.get('input_text_and_control_tokens', 0) for e in events if e['name'] == name),
                ovis_image_atom_tokens=sum(e.get('ovis_image_atom_tokens', 0) for e in events if e['name'] == name),
                ovis_visual_indicator_tokens=sum(e.get('ovis_visual_indicator_tokens', 0) for e in events if e['name'] == name),
                kosmos_image_embedding_positions=sum(e.get('kosmos_image_embedding_positions', 0) for e in events if e['name'] == name),
                regional_mask_positions=sum(e.get('regional_mask_positions', 0) for e in events if e['name'] == name),
                regional_depth_positions=sum(e.get('regional_depth_positions', 0) for e in events if e['name'] == name),
                effective_input_embedding_tokens=sum(e.get('effective_input_embedding_tokens', 0) for e in events if e['name'] == name))
                for name in names})
        for r in rows:
            stages[r['method']].setdefault(r['id'], {})[r['stage']] = r['sample_seconds']
    result['composed_online_stage_sums'] = {}
    required = {'spatialpin': {'discover', 'prepare', 'evaluate'}, 'spatialrgpt': {'regions', 'evaluate'}}
    for method, ids in stages.items():
        needed = required.get(method)
        if needed:
            totals = [sum(v[k] for k in needed) for v in ids.values() if needed <= set(v)]
            result['composed_online_stage_sums'][method] = dict(
                valid_for_reporting=method not in invalid,
                statistic=stats(totals), scope='sum of corresponding warm stage wall times; excludes model reload, process launch and transfers')
    output = args.directory / 'cost_summary.json'
    result['measured_online_requests'] = {
        p.stem: json.loads(p.read_text(encoding='utf-8')) for p in args.directory.glob('*_online.summary.json')}
    for value in result['measured_online_requests'].values():
        value['valid_for_reporting'] = value.get('method') not in invalid
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(output)
    for key, value in result['stages'].items():
        print(key, json.dumps(dict(samples=value['samples'], wall=value['wall'], statuses=value['statuses'])))


if __name__ == '__main__':
    main()

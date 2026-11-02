import json,sys
from collections import Counter
from pathlib import Path
from protocol import VERSION,GREEDY

def verify(method, data, out):
    data=Path(data);out=Path(out);items=json.loads(data.read_text());ids=[str(x['id']) for x in items]
    pred_path=out/({'ovis':'ovis_native_predictions.jsonl','bydeway':'bydeway_predictions.jsonl'}.get(method,method+'/predictions.jsonl'))
    rows=[json.loads(x) for x in pred_path.read_text().splitlines() if x.strip()]
    actual=[str(r['id']) for r in rows]
    if Counter(actual)!=Counter(ids) or len(set(actual))!=len(actual):raise RuntimeError('Missing/duplicate/unexpected predictions')
    formal=[r for r in rows if not str(r['id']).startswith('warmup_')]
    status=Counter(r.get('status') for r in formal)
    if any(r.get('status')!='ok' for r in rows):raise RuntimeError('Engineering failure; inspect all attempts before next method')
    expected={str(x['id']):x for x in items}
    valid=correct=0
    for r in formal:
        p=r.get('prediction',r.get('pred_answer'));row=expected[str(r['id'])]
        legal={str(i) for i in range(1,5) if str(row.get('Option_'+str(i),'') or '').strip()}
        valid+=p in legal;correct+=p in legal and p==str(row['Answer'])
    if method!='spatialrgpt':
        audit=[json.loads(x) for x in (out/'effective_decoding_calls.jsonl').read_text().splitlines()]
        for a in audit:
            if a['status']!='ok' or a['effective']!=GREEDY:raise RuntimeError('Internal decoding audit failure')
        for item_id in ids:
            req=[a for a in audit if a['context']['id']==item_id]
            stages={a['context']['stage'] for a in req}
            if stages!=({'discover','evaluate'} if method=='spatialpin' else {'evaluate'}):
                raise RuntimeError('Missing inner audit for '+item_id)
        specials=[json.loads(x) for x in (out/'effective_special_tokens.jsonl').read_text().splitlines()]
        if len(specials)!=len(audit) or any(s['pad']!=151645 or s['eos'] not in [151645,[151645]] for s in specials):
            raise RuntimeError('Effective pad/EOS audit missing or changed')
    report=dict(protocol=VERSION,method=method,planned=len(formal),engineering_success=status.get('ok',0),
        valid_answers=valid,unparsed_or_invalid=len(formal)-valid,correct=correct,accuracy_percent=100*correct/len(formal),
        warmups_excluded=len(rows)-len(formal),denominator_policy='all planned formal samples; invalid answers wrong')
    (out/'protocol_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':verify(*sys.argv[1:])

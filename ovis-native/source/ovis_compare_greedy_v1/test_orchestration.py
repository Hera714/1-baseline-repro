"""CPU simulation of the SAME coordinator in smoke/full/failure modes; no models."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import suite,assets

for mode in ['smoke','full']:
    with tempfile.TemporaryDirectory() as d:
        calls=[]
        with patch.object(sys,'argv',['suite.py',mode,'--output',str(Path(d)/'run')]),patch.object(assets,'check',lambda:None),patch.object(suite,'run_remote',lambda m,args:calls.append((m,args[0]))):
            suite.main()
        assert calls==[(m,'check') for m in ['ovis','spatialpin','bydeway','spatialrgpt']]+[(m,'worker') for m in ['ovis','spatialpin','bydeway','spatialrgpt']],calls
        assert (Path(d)/'run/ALL_COMPLETED').exists()
with tempfile.TemporaryDirectory() as d:
    calls=[]
    def fail(m,args):
        calls.append((m,args[0]))
        if m=='spatialpin' and args[0]=='worker':raise RuntimeError('simulated failure')
    with patch.object(sys,'argv',['suite.py','full','--output',str(Path(d)/'run')]),patch.object(assets,'check',lambda:None),patch.object(suite,'run_remote',fail):
        try:suite.main()
        except RuntimeError:pass
        else:raise AssertionError('Failure ignored')
    assert ('bydeway','worker') not in calls and ('spatialrgpt','worker') not in calls
    assert not (Path(d)/'run/ALL_COMPLETED').exists()
print('ORCHESTRATION CPU PASS: same serial coordinator; smoke/full ordering; fail-fast; no overlap; no GPU/model run.')

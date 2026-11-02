"""One serial coordinator; check=CPU contract tests, smoke=real 8-item GPU run, full=1402-item GPU run."""
import argparse,fcntl,hashlib,json,os,shlex,signal,socket,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HOSTS={'spatialpin':('autodl-pro-7919ba5077f0',29937),'bydeway':('autodl-pro-7919ba5077f0',29937),
       'ovis':('autodl-pro-7919ba5077f0',29937),'spatialrgpt':('autodl-pro-791817ed3958',44571)}
ENV={'spatialpin':'/root/ovis-compare-greedy-v1-env/bin/python','ovis':'/root/ovis-compare-greedy-v1-env/bin/python',
     'bydeway':'/root/ovis-compare-greedy-v1-env/bin/python','spatialrgpt':'/root/autodl-fs/baselines/envs/spatialrgpt/bin/python'}
BASEPY='/root/miniconda3/bin/python'

def run_remote(method,args):
    host,port=HOSTS[method]
    cmd=[BASEPY,str(ROOT/'suite.py')]+args+['--method',method]
    if socket.gethostname()==host:
        subprocess.run(cmd,check=True)
    else:
        subprocess.run(['ssh','-o','ControlMaster=auto','-o','ControlPersist=48h','-o','ControlPath='+str(Path(os.environ.get('COMPARE_CONTROL_DIR','/tmp/ovis_compare_ssh'))/'%C'),'-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'known_hosts'),
            '-p',str(port),'root@connect.westd.seetacloud.com',shlex.join(cmd)],check=True)

def gpu_pids():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(x.strip()) for x in p.stdout.splitlines() if x.strip().isdigit()}

def descendants(pid):
    result={pid};parents={}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            text=p.read_text();parents[int(p.parent.name)]=int(text[text.rfind(')')+2:].split()[1])
        except (OSError,ValueError,IndexError):pass
    while True:
        new={p for p,pp in parents.items() if pp in result}
        if new<=result:return result
        result|=new

def cpu_check(method):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    import ast
    assert socket.gethostname()==HOSTS[method][0], 'Wrong designated server'
    for p in list(ROOT.glob('*.py'))+list((ROOT/'sources').glob('*.py')):ast.parse(p.read_text(),filename=str(p))
    subprocess.run(['bash','-n',str(ROOT/'run.sh')],check=True)
    code='import torch, transformers; print("ENV",torch.__version__,transformers.__version__)'
    subprocess.run([ENV[method],'-c',code],check=True)
    for env in ([ENV[method],'/root/autodl-fs/baselines/envs/spatialpin/bin/python'] if method=='spatialpin' else [ENV[method]]):
        subprocess.run([env,str(ROOT/'test_protocol.py')],check=True)
    if method=='spatialrgpt':
        subprocess.run(['/root/cost-detector-env/bin/python','-c','import torch,transformers; print("DETECTOR IMPORT OK")'],check=True)
    if method=='bydeway':
        code = "import sys,importlib.util; sys.path[:0]=['/root/autodl-fs/baselines/bydeway/ByDeWay-main','/root/autodl-fs/baselines/bydeway/ByDeWay-main/Depth-Anything-V2']; s=importlib.util.spec_from_file_location('checked_bydeway',%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('BYDEWAY IMPORT OK',m.MAX_NEW_TOKENS,m.THINKING_BUDGET)" % str(ROOT/'bydeway_adapter.py')
        subprocess.run([ENV[method],'-c',code],check=True)
    if method=='spatialpin':
        env=dict(os.environ,PYTHONPATH='/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin:/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin/main:/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin/export',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
        subprocess.run(['/root/autodl-fs/baselines/envs/spatialpin/bin/python','-c','from lang_segment_anything.lang_sam import LangSAM; from Depth_Anything.depth_anything.dpt import DepthAnything; print("PERCEPTION IMPORT OK")'],env=env,check=True)
    # Help commands are the same executable entry points as the real worker, without loading weights.
    subprocess.run([ENV[method],str(ROOT/'run_cost.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
    print('CPU CHECK PASSED',method,'(not a real GPU inference guarantee)',flush=True)

def worker(method,mode,out):
    if socket.gethostname()!=HOSTS[method][0]:raise RuntimeError('Wrong machine')
    import assets
    assets.check()
    if gpu_pids():raise RuntimeError('GPU busy: stop other inference/training before cost measurement')
    data=ROOT/'manifests'/('smoke.json' if mode=='smoke' else 'full.json')
    out=Path(out)/method
    if out.exists():raise RuntimeError('Fresh output required: '+str(out))
    out.mkdir(parents=True)
    (out/'launch_manifest.json').write_text(json.dumps(dict(protocol='ovis_compare_greedy_v1',method=method,
        mode=mode,hostname=socket.gethostname(),data=str(data),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        asset_lock_sha256=hashlib.sha256((ROOT/'assets.lock.json').read_bytes()).hexdigest(),time=time.time()),indent=2))
    env=dict(os.environ,COST_DATA=str(data),COST_RUN=str(out),PYTHONUNBUFFERED='1')
    stage=method+'-online' if method in ['spatialpin','spatialrgpt'] else method
    cmd=['bash',str(ROOT/'run.sh'),stage]
    with (out/'coordinator.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
        contamination=[];stop=threading.Event()
        def monitor():
            while not stop.wait(.5):
                try:
                    foreign=gpu_pids()-descendants(os.getpid())
                    if foreign:
                        contamination.append(dict(time=time.time(),pids=sorted(foreign)))
                        (out/'gpu_contamination.json').write_text(json.dumps(contamination,indent=2))
                        os.killpg(proc.pid,signal.SIGTERM);return
                except Exception as exc:
                    contamination.append(dict(time=time.time(),monitor_error=repr(exc)))
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            rc=proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise
        finally:stop.set();thread.join()
    if contamination:raise RuntimeError('Foreign GPU process or monitoring failure; run invalid; next method not launched')
    if rc:raise RuntimeError('Method failed; next method not launched: '+str(rc))
    subprocess.run([BASEPY,str(ROOT/'verify_results.py'),method,str(data),str(out)],check=True)
    subprocess.run(['bash',str(ROOT/'run.sh'),'summary'],env=env,check=True)
    if gpu_pids():raise RuntimeError('GPU workers remain alive; refusing next method')
    (out/'COMPLETED').write_text('Verified complete; all workers exited\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['check','smoke','full','worker'])
    ap.add_argument('--method',choices=list(HOSTS))
    ap.add_argument('--methods',default='ovis,spatialpin,bydeway,spatialrgpt')
    ap.add_argument('--run-mode',choices=['smoke','full'])
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.mode=='worker':return worker(a.method,a.run_mode,a.output)
    if a.mode=='check' and a.method:return cpu_check(a.method)
    control=Path('/tmp')/('ovis_compare_ssh_'+str(os.getuid()))
    control.mkdir(mode=0o700,exist_ok=True)
    os.environ['COMPARE_CONTROL_DIR']=str(control)
    methods=a.methods.split(',')
    if len(set(methods))!=len(methods) or any(m not in HOSTS for m in methods):ap.error('Invalid methods')
    if a.mode=='check':
        for m in methods:run_remote(m,['check'])
        return
    if a.output is None:ap.error('--output required; choose a NEW directory')
    a.output=a.output.resolve()
    lock=(ROOT/'suite.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.output.exists():raise RuntimeError('No resume/mixed configuration: choose new --output')
    # All checks must pass before the first model starts; freezes detect edits between methods.
    import assets
    assets.check()
    for m in methods:run_remote(m,['check'])
    a.output.mkdir(parents=True)
    (a.output/'plan.json').write_text(json.dumps(dict(mode=a.mode,methods=methods,serial=True),indent=2))
    for m in methods:
        print('=== SERIAL START',m,'===',flush=True)
        run_remote(m,['worker','--run-mode',a.mode,'--output',str(a.output)])
        print('=== VERIFIED END',m,'===',flush=True)
    (a.output/'ALL_COMPLETED').write_text('All methods complete; no parallel model runs\n')
if __name__=='__main__':main()

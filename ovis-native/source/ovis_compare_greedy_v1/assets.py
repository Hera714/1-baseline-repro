import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL=Path('/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def inputs():
    paths=list(MODEL.glob('*'))
    models=Path('/root/autodl-fs/baselines/models')
    paths += [p for p in (models/'bydeway').rglob('*') if p.is_file() and '.cache' not in p.parts]
    paths += list((models/'SpatialRGPT-VILA1.5-8B').glob('*.bin'))+list((models/'SpatialRGPT-VILA1.5-8B').glob('*.safetensors'))
    paths += list((models/'SpatialRGPT-VILA1.5-8B').glob('*.json'))
    for sub in ['Depth-Anything/checkpoints/depth_anything_vitl14.pth','grounding-dino-tiny']:
        p=models/sub
        paths += list(p.glob('*')) if p.is_dir() else [p]
    spin=Path('/root/autodl-fs/baselines/spatialpin/SpatialPIN/spatialpin/main/checkpoints')
    paths+=list(spin.rglob('*.pth'))+list(spin.rglob('*.py'))
    paths+=list((ROOT/'generalization').glob('*.py'))+list((ROOT/'frozen_kar').rglob('*.py'))
    paths+=list(ROOT.glob('*.py'))+list((ROOT/'sources').glob('*.py'))+list(ROOT.glob('*.sh'))+list((ROOT/'manifests').glob('*.json'))
    for source in ['/root/autodl-fs/baselines/bydeway/ByDeWay-main/src','/root/autodl-fs/baselines/bydeway/ByDeWay-main/Depth-Anything-V2/depth_anything_v2']:
        paths+=list(Path(source).rglob('*.py'))
    return sorted(set(p.resolve() for p in paths if p.is_file()))
def freeze():
    target=ROOT/'assets.lock.json'
    if target.exists():raise RuntimeError('Lock already exists; do not silently replace a frozen version')
    records=[]
    for p in inputs():
        st=p.stat();records.append(dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p)))
        print('hashed',p,flush=True)
    data=json.loads((ROOT/'manifests/full.json').read_text());images=[]
    for row in data:
        if str(row['id']).startswith('warmup_'):continue
        p=Path('/root/autodl-fs/img')/row['Image_Filename'];st=p.stat()
        images.append(dict(id=str(row['id']),path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p)))
    target.write_text(json.dumps(dict(protocol='ovis_compare_greedy_v1',files=records,images=images),indent=2))
    print('FROZEN',len(records),'files',len(images),'images',flush=True)
def check(deep=False):
    lock=json.loads((ROOT/'assets.lock.json').read_text())
    for r in lock['files']+lock['images']:
        p=Path(r['path']);st=p.stat()
        if st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']:
            raise RuntimeError('Frozen asset changed: '+str(p))
        if deep and sha(p)!=r['sha256']:raise RuntimeError('SHA256 changed: '+str(p))
    print('ASSETS OK: sizes/mtime match full SHA256 lock'+('; full SHA256 reverified' if deep else '; use --deep to recompute all hashes'))
if __name__=='__main__':
    freeze() if '--freeze' in sys.argv else check('--deep' in sys.argv)

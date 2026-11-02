"""Use existing DINOv2 files without Torch Hub's preliminary GitHub lookup."""
import os
from pathlib import Path
from urllib.parse import urlparse


def install():
    import torch
    if getattr(torch.hub, '_cost_offline_dino', False):
        return
    original_load = torch.hub.load
    original_state = torch.hub.load_state_dict_from_url

    def local_load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).split(':', 1)[0] == 'facebookresearch/dinov2':
            cached = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
            bundled = Path('/root/autodl-fs/baselines/models/Depth-Anything/torchhub/facebookresearch_dinov2_main')
            directory = cached if (cached / 'hubconf.py').is_file() else bundled
            if not (directory / 'hubconf.py').is_file():
                raise FileNotFoundError('Local DINOv2 source not found: ' + str(directory))
            repo_or_dir = str(directory)
            kwargs['source'] = 'local'
            print('Offline DINOv2 source: ' + repo_or_dir, flush=True)
        return original_load(repo_or_dir, model, *args, **kwargs)

    def local_state(url, model_dir=None, *args, **kwargs):
        if 'dinov2' in str(url):
            directory = Path(model_dir) if model_dir else Path(torch.hub.get_dir()) / 'checkpoints'
            filename = kwargs.get('file_name') or os.path.basename(urlparse(str(url)).path)
            if not (directory / filename).is_file():
                raise FileNotFoundError('Offline DINOv2 initialization checkpoint missing: ' + str(directory / filename))
        return original_state(url, model_dir, *args, **kwargs)

    torch.hub.load = local_load
    torch.hub.load_state_dict_from_url = local_state
    torch.hub._cost_offline_dino = True

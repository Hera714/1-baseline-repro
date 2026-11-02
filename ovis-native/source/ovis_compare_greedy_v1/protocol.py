"""Frozen ovis_compare_greedy_v1: explicit outer settings + resolved inner audit."""
import copy
import hashlib
import inspect
import importlib.metadata as metadata
import json
import os
import random
import time
from pathlib import Path
from canonical_evaluation import extract_answer

VERSION = 'ovis_compare_greedy_v1'
GREEDY = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
              repetition_penalty=1.0, num_beams=1)
MIN_PIXELS, MAX_PIXELS = 448*448, 1792*1792
MODEL = Path('/root/autodl-fs/modelscope_cache/AIDC-AI/Ovis2___5-9B')

class ProtocolError(RuntimeError):
    pass

def reset_seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

def format_options(item):
    return '\n'.join(str(i)+'. '+str(item['Option_'+str(i)]) for i in range(1,5)
                     if str(item.get('Option_'+str(i), '') or '').strip())

def parse_options(raw, options):
    pred=extract_answer(raw or '')
    valid={str(i) for i,opt in enumerate(options,1) if str(opt or '').strip()}
    return (pred, VERSION) if pred in valid else (None, 'unparsed_or_invalid_option')

def parse_item(raw, item):
    return parse_options(raw,[item.get('Option_'+str(i),'') for i in range(1,5)])

def write_event(out, filename, event):
    out.mkdir(parents=True,exist_ok=True)
    with (out/filename).open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False)+'\n')

def resolved(llm, kwargs):
    cfg,_=llm._prepare_generation_config(copy.deepcopy(kwargs.get('generation_config')),
                 **{k:v for k,v in kwargs.items() if k!='generation_config'})
    return cfg

def install_ovis(model, out, context=None):
    if getattr(model,'_compare_protocol_installed',False): return
    import torch, transformers
    out=Path(out); context=context or (lambda: {})
    llm=model.llm
    expected_packages={'transformers':'4.51.3','modelscope':'1.35.1','flash-attn':'2.7.4.post1','numpy':'1.26.4','Pillow':'11.3.0','timm':'0.9.16','tokenizers':'0.21.4','torchvision':'0.20.1','huggingface-hub':'0.36.2','accelerate':'1.13.0','safetensors':'0.8.0'}
    versions={}
    for name in expected_packages:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    if type(model).__name__=='Ovis2_5':
        if versions!=expected_packages or torch.__version__!='2.5.1+cu124':
            raise ProtocolError('Ovis execution environment changed: '+str(versions)+' torch='+torch.__version__)
    initial={k:getattr(llm.generation_config,k) for k in GREEDY}
    if initial != GREEDY: raise ProtocolError('Unexpected internal defaults: '+str(initial))
    source=Path(inspect.getfile(type(model)))
    expected=MODEL/'modeling_ovis2_5.py'
    expected_sha=hashlib.sha256(expected.read_bytes()).hexdigest()
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!=expected_sha:
        raise ProtocolError('Loaded custom code differs from frozen local Ovis source: '+str(source))
    snapshot=dict(protocol=VERSION,packages=versions,outer_defaults={k:getattr(model.generation_config,k) for k in GREEDY},
        inner_defaults=initial,explicit=GREEDY,torch=torch.__version__,transformers=transformers.__version__,
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model=str(MODEL),seed=42,seed_scope='each item and each outer Ovis request',
        min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,parameter_dtype=str(next(model.parameters()).dtype),
        attention_implementation=getattr(llm.config,'_attn_implementation',None),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    write_event(out,'runtime_decoding.jsonl',dict(context=context(),**snapshot))
    outer_generate,inner_generate,preprocess=model.generate,llm.generate,model.preprocess_inputs
    state=dict(request=None,preprocess=None)

    def prep(*args,**kwargs):
        bound=inspect.signature(preprocess).bind(*args,**kwargs);bound.apply_defaults()
        a=bound.arguments
        if a.get('min_pixels')!=MIN_PIXELS or a.get('max_pixels')!=MAX_PIXELS:
            raise ProtocolError('Ovis pixel budget mismatch')
        if a.get('add_generation_prompt') is not True:
            raise ProtocolError('Missing generation prompt')
        visible=[]
        for msg in a['messages']:
            if msg.get('role')!='user': raise ProtocolError('Unexpected message role')
            content=[]
            for part in msg['content']:
                if part.get('type')=='image':
                    im=part['image']
                    rgb=im.convert('RGB')
                    content.append(dict(type='image',size=list(rgb.size),mode=im.mode,
                          rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                elif part.get('type')=='text':
                    content.append(dict(type='text',text=part['text'],sha256=hashlib.sha256(part['text'].encode()).hexdigest()))
                else: raise ProtocolError('Unexpected multimodal content')
            visible.append(dict(role=msg['role'],content=content))
        result=preprocess(*args,**kwargs)
        ids,pixels,grid=result
        state['preprocess']=dict(enable_thinking=a['enable_thinking'])
        write_event(out,'visible_inputs.jsonl',dict(context=context(),messages=visible,
             min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,enable_thinking=a['enable_thinking'],
             input_tokens=ids.numel(),input_ids=ids.detach().cpu().tolist(),
             pixel_dtype=str(pixels.dtype) if pixels is not None else None,
             pixel_shape=list(pixels.shape) if pixels is not None else None,
             grid_thws=grid.detach().cpu().tolist() if grid is not None else None))
        return result

    def outer(*args,**kwargs):
        reset_seed()
        for k,v in GREEDY.items():
            if k in kwargs and kwargs[k]!=v: raise ProtocolError('Conflicting '+k)
            kwargs[k]=v
        stage=context().get('stage','evaluate')
        discovering=stage=='discover'
        expected_max=1024 if discovering else 2048
        if kwargs.get('max_new_tokens')!=expected_max:
            raise ProtocolError('Output budget mismatch: '+str(kwargs.get('max_new_tokens')))
        thinking=bool(kwargs.get('enable_thinking',False))
        if thinking != (not discovering): raise ProtocolError('Thinking stage mismatch')
        if state['preprocess'] is None or state['preprocess']['enable_thinking']!=thinking:
            raise ProtocolError('Preprocess/generate thinking mismatch')
        if not discovering and (kwargs.get('enable_thinking_budget') is not True or kwargs.get('thinking_budget')!=1536):
            raise ProtocolError('Final thinking budget must be 1536')
        if discovering and kwargs.get('enable_thinking_budget',False):
            raise ProtocolError('Discovery budget must be off')
        state['request']=dict(context=context(),max_new_tokens=expected_max,enable_thinking=thinking,
                              thinking_budget=kwargs.get('thinking_budget'),inner_calls=0)
        with torch.no_grad(): return outer_generate(*args,**kwargs)

    def inner(*args,**kwargs):
        cfg=resolved(llm,kwargs)
        values={k:getattr(cfg,k) for k in GREEDY}
        if values!=GREEDY: raise ProtocolError('Inner effective greedy mismatch: '+str(values))
        if cfg.eos_token_id!=151645 or cfg.bos_token_id!=151643 or cfg.num_return_sequences!=1 or not cfg.use_cache:
            raise ProtocolError('EOS/BOS/cache/return count changed')
        if cfg.pad_token_id not in [None,151645]: raise ProtocolError('Unexpected pad')
        req=state['request']
        if req is None: raise ProtocolError('Inner LLM bypassed outer Ovis request')
        req['inner_calls']+=1
        row=dict(protocol=VERSION,context=context(),request=dict(req),effective=values,
            max_new_tokens=cfg.max_new_tokens,eos_token_id=cfg.eos_token_id,bos_token_id=cfg.bos_token_id,
            pad_token_id_before_fallback=cfg.pad_token_id,
            pad_token_id_expected_after_fallback=151645,num_return_sequences=cfg.num_return_sequences,
            use_cache=cfg.use_cache,length_penalty=cfg.length_penalty,no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            time=time.time(),inputs_embeds_shape=list(kwargs['inputs_embeds'].shape) if kwargs.get('inputs_embeds') is not None else None)
        if cfg.max_new_tokens<=0: raise ProtocolError('Nonpositive continuation budget')
        start=time.perf_counter()
        try:
            result=inner_generate(*args,**kwargs)
            seq=result.sequences if hasattr(result,'sequences') else result
            row.update(status='ok',output_tokens=int(seq[0].numel()),output_token_ids=seq[0].detach().cpu().tolist(),
                       output_at_token_limit=int(seq[0].numel())>=cfg.max_new_tokens)
            return result
        except BaseException as exc:
            row.update(status='error',error=repr(exc));raise
        finally:
            row['seconds']=time.perf_counter()-start
            write_event(out,'effective_decoding_calls.jsonl',row)
    # Capture the pad/EOS tensors after Transformers resolves fallback.
    if hasattr(llm,'_prepare_special_tokens'):
        special=llm._prepare_special_tokens
        def prepare_special(*args,**kwargs):
            result=special(*args,**kwargs)
            cfg=args[0] if args else kwargs['generation_config']
            def val(name):
                t=getattr(cfg,name,None)
                return t.detach().cpu().tolist() if torch.is_tensor(t) else t
            write_event(out,'effective_special_tokens.jsonl',dict(context=context(),
                pad=val('_pad_token_tensor'),eos=val('_eos_token_tensor'),bos=val('_bos_token_tensor')))
            return result
        llm._prepare_special_tokens=prepare_special
    model.preprocess_inputs=prep;model.generate=outer;llm.generate=inner
    model._compare_protocol_installed=True

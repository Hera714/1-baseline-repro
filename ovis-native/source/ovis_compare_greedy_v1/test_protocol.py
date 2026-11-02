"""CPU contract test of real custom Ovis.generate body; fake decoder, no weights/GPU."""
import ast,copy,json,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from transformers import GenerationConfig
import protocol

class FakeLLM:
    def __init__(self):
        self.generation_config=GenerationConfig(bos_token_id=151643,eos_token_id=151645,**protocol.GREEDY)
        self.config=types.SimpleNamespace(_attn_implementation='flash_attention_2')
        self.calls=0
    def _prepare_generation_config(self,generation_config=None,**kwargs):
        cfg=copy.deepcopy(generation_config or self.generation_config)
        remaining=cfg.update(**kwargs)
        return cfg,remaining
    def _prepare_special_tokens(self,cfg,**kwargs):
        cfg._pad_token_tensor=torch.tensor(cfg.pad_token_id or cfg.eos_token_id)
        cfg._eos_token_tensor=torch.tensor([cfg.eos_token_id]);cfg._bos_token_tensor=torch.tensor(cfg.bos_token_id)
    def generate(self,**kwargs):
        cfg,_=self._prepare_generation_config(**kwargs)
        self._prepare_special_tokens(cfg)
        self.calls+=1
        # First pass exhausts mock thinking without EOS; actual Ovis must continue.
        return torch.tensor([[77]]) if self.calls%2 else torch.tensor([[151645]])

class FakeOvis:
    def __init__(self,fn):
        self.llm=FakeLLM();self.generation_config=GenerationConfig(do_sample=True,temperature=.6,top_p=.95,top_k=20,repetition_penalty=1.05)
        self.text_tokenizer=types.SimpleNamespace(pad_token_id=151643)
        # Support actual source's forced-thinking stop tokenization.
        class Tokenizer:
            pad_token_id=151643
            def __call__(self,*a,**k):return types.SimpleNamespace(input_ids=torch.tensor([[151668]]))
        self.text_tokenizer=Tokenizer();self.generate=types.MethodType(fn,self)
    def parameters(self):return iter([torch.zeros(1,dtype=torch.bfloat16)])
    def merge_multimodal(self,input_ids,**kwargs):return torch.zeros(1,input_ids.shape[1],4)
    def preprocess_inputs(self,messages,min_pixels=448*448,max_pixels=1792*1792,add_generation_prompt=True,enable_thinking=False):
        return torch.tensor([[1,2,3]]),None,None

source=protocol.MODEL/'modeling_ovis2_5.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Ovis2_5')
fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate'))
fn.returns=None
for a in fn.args.args: a.annotation=None
namespace={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
with tempfile.TemporaryDirectory() as directory:
    out=Path(directory);model=FakeOvis(namespace['generate'])
    with patch.object(protocol.inspect,'getfile',return_value=str(source)):
        protocol.install_ovis(model,out,lambda:dict(id='cpu_contract',stage='evaluate'))
    model.preprocess_inputs(messages=[dict(role='user',content=[dict(type='image',image=Image.new('RGB',(64,32))),dict(type='text',text='test')])],enable_thinking=True)
    result=model.generate(inputs=torch.tensor([[1,2,3]]),max_new_tokens=2048,enable_thinking=True,enable_thinking_budget=True,thinking_budget=1536)
    rows=[json.loads(x) for x in (out/'effective_decoding_calls.jsonl').read_text().splitlines()]
    assert len(rows)==2 and [r['max_new_tokens'] for r in rows]==[1536,2046],rows
    assert all(r['effective']==protocol.GREEDY for r in rows)
    for bad in [dict(do_sample=True),dict(max_new_tokens=3072),dict(thinking_budget=2048)]:
        kw=dict(inputs=torch.tensor([[1,2,3]]),max_new_tokens=2048,enable_thinking=True,enable_thinking_budget=True,thinking_budget=1536);kw.update(bad)
        try:model.generate(**kw)
        except protocol.ProtocolError:pass
        else:raise AssertionError('Conflicting settings accepted: '+str(bad))
    assert protocol.parse_options('<think>1</think><answer>2</answer><answer>3</answer>',['a','b','c','d'])[0]=='3'
    assert protocol.parse_options('<think>1',['a','b','c','d'])[0] is None
    assert protocol.parse_options('<answer>4</answer>',['a','b','c',''])[0] is None
    assert protocol.parse_options('<think>1</think>Option 2',['a','b','c','d'])[0]=='2'
    assert protocol.format_options(dict(Option_1='a',Option_2='b',Option_3='c',Option_4=''))=='1. a\n2. b\n3. c'
print('CPU CONTRACT PASS: actual custom generate -> two internal calls; six resolved fields; budgets; EOS/pad audit; refusal; canonical parser. NOT real GPU smoke.')

"""Authenticated readiness and five bounded API checks; never print credentials."""
import argparse,json,time,urllib.request,urllib.error
from pathlib import Path

def request(base,key,path,body=None,timeout=30):
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    req=urllib.request.Request(base.rstrip('/')+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout)

def wait_ready(base,key,model,timeout=1800):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            with request(base,key,'/v1/models',timeout=5) as r:data=json.load(r)
            if model in [v['id'] for v in data.get('data',[])]:return
        except (OSError,ValueError):pass
        time.sleep(3)
    raise TimeoutError('Authenticated model readiness timed out')

def verify(base,key,model):
    checks=[]
    with request(base,key,'/v1/models') as r:data=json.load(r)
    assert model in [v['id'] for v in data['data']];checks.append('model_listing')
    body={'model':model,'messages':[{'role':'user','content':'Reply with exactly READY.'}],
        'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
    with request(base,key,'/v1/chat/completions',body,120) as r:data=json.load(r)
    assert data['choices'][0]['message']['content'].strip() and data['usage']['completion_tokens']>0
    checks.append('chat_generation')
    with request(base,key,'/v1/completions',{'model':model,'prompt':'def add(a, b):\n    return',
                 'temperature':0,'max_tokens':16},120) as r:data=json.load(r)
    assert data['choices'][0]['text'] and data['usage']['completion_tokens']>0;checks.append('completion_generation')
    text=[];done=False
    with request(base,key,'/v1/chat/completions',dict(body,stream=True),120) as r:
        for raw in r:
            if not raw.startswith(b'data:'):continue
            value=raw[5:].strip()
            if value==b'[DONE]':done=True;break
            packet=json.loads(value)
            for choice in packet.get('choices',[]):text.append(choice.get('delta',{}).get('content') or '')
    assert done and ''.join(text).strip();checks.append('streaming_generation')
    try:
        with request(base,'invalid-release-check-key','/v1/chat/completions',body,10):pass
    except urllib.error.HTTPError as error:assert error.code in (401,403)
    else:raise AssertionError('Invalid API key was accepted')
    checks.append('invalid_key_rejected')
    return {'passed':len(checks),'checks':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:19890')
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--model',default='DeepSeek-V4.1-Flash-1017')
    p.add_argument('--wait',type=int,default=0);a=p.parse_args();key=a.key_file.read_text().strip()
    if a.wait:wait_ready(a.base,key,a.model,a.wait)
    print(json.dumps(verify(a.base,key,a.model)))

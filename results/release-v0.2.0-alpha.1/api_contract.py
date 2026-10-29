"""Additional coding-client API checks; preserve every request and response."""
import argparse,json,pathlib,re,time,urllib.request

p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--key-file',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
key=a.key_file.read_text().strip();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
schema={'type':'object','properties':{'ready':{'type':'boolean'},'count':{'type':'integer'}},'required':['ready','count'],'additionalProperties':False}
tool={'type':'function','function':{'name':'record_status','description':'Record readiness and item count.','parameters':schema}}
requests=[
 ('json-schema','Return ready=true and count=3.',{'response_format':{'type':'json_schema','json_schema':{'name':'status','strict':True,'schema':schema}}}),
 ('tool-call','Record readiness as true with 3 items using the provided tool.',{'tools':[tool],'tool_choice':'required'}),
 ('regex','Return the date 2026-10-28.',{'guided_regex':r'\d{4}-\d{2}-\d{2}'}),
]
rows=[]
for name,prompt,extra in requests:
    body={'model':'DeepSeek-V4.1-Flash-1017','messages':[{'role':'user','content':prompt}], 'temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False},**extra}
    req=urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    started=time.monotonic()
    with opener.open(req,timeout=180) as response:data=json.load(response)
    choice=data['choices'][0];message=choice['message']
    if name=='json-schema':passed=json.loads(message['content'])=={'ready':True,'count':3}
    elif name=='tool-call':
        calls=message.get('tool_calls') or []
        passed=choice['finish_reason']=='tool_calls' and len(calls)==1 and calls[0]['function']['name']=='record_status' and json.loads(calls[0]['function']['arguments'])=={'ready':True,'count':3}
    else:passed=re.fullmatch(r'\d{4}-\d{2}-\d{2}',message['content'].strip()) is not None
    (a.output/(name+'.json')).write_text(json.dumps({'request':body,'response':data,'passed':passed},indent=2))
    rows.append({'name':name,'passed':passed,'seconds':time.monotonic()-started});print(json.dumps(rows[-1]),flush=True)
(a.output/'summary.json').write_text(json.dumps(rows,indent=2))
assert all(v['passed'] for v in rows),rows

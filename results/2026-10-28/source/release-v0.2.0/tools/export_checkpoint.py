"""Byte-exact export of selected packed experts. Never writes to the source.
Pruning removes experts and changes the model function; this is not quality-lossless.

Selection JSON: {"0": [old expert IDs in new-ID order], ..., "39": [...]}
All backbone layers must retain the same expert count. MTP, shared experts,
attention, embeddings and Engram are preserved. No dequantization occurs.
"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import time

EXPERT = re.compile(r'^(layers\.(\d+)\.ffn\.experts\.)(\d+)(?=\.|$)')
GATE = re.compile(r'^layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$')

def header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return 8+n,json.loads(f.read(n))

def export(source, destination, selections):
    source=source.resolve(); destination=destination.resolve()
    if destination==source or source in destination.parents:
        raise ValueError('Destination must be outside source checkpoint.')
    if destination.exists(): raise FileExistsError(destination)
    config=json.loads((source/'config.json').read_text())
    tc=config.get('text_config',config)
    layers=tc['num_hidden_layers']; original=tc['n_routed_experts']
    assert set(selections)=={str(i) for i in range(layers)}
    counts={len(v) for v in selections.values()};assert len(counts)==1
    kept=counts.pop();assert tc['num_experts_per_tok']<=kept<=original
    for values in selections.values():
        assert len(set(values))==kept and all(isinstance(x,int) and 0<=x<original for x in values)
    maps={k:{old:new for new,old in enumerate(values)} for k,values in selections.items()}
    def rename(name):
        if not isinstance(name,str):return name
        match=EXPERT.match(name)
        if not match:return name
        new=maps[match[2]].get(int(match[3]))
        if new is None:return None
        return match[1]+str(new)+name[match.end():]
    def metadata(value,parent_key=None):
        if isinstance(value,dict):
            result={rename(k):metadata(v,k) for k,v in value.items() if rename(k) is not None}
            if isinstance(parent_key,str) and GATE.match(parent_key) and 'shape' in result:
                assert result['shape'][0]==original
                result['shape'][0]=kept
                if 'n_bytes' in result:
                    assert result['n_bytes']%original==0
                    result['n_bytes']=result['n_bytes']//original*kept
            return result
        if isinstance(value,list): return [metadata(v) for v in value if not isinstance(v,str) or rename(v) is not None]
        return rename(value) if isinstance(value,str) else value
    temp=destination.with_name(destination.name+'.partial')
    if temp.exists():raise FileExistsError(temp)
    temp.mkdir(parents=True)
    index={}; manifest=[]; total=0; started=time.monotonic()
    for src in sorted(source.glob('*.safetensors')):
        base,old=header(src); new={}; spans=[]; offset=0
        if '__metadata__' in old:new['__metadata__']=old['__metadata__']
        for name,entry in old.items():
            if name=='__metadata__':continue
            name2=rename(name)
            if name2 is None:continue
            e=copy.deepcopy(entry); lo,hi=e['data_offsets']; ranges=[(base+lo,hi-lo)]
            match=GATE.match(name)
            if match:
                assert e['shape'][0]==original and (hi-lo)%original==0
                row=(hi-lo)//original
                ranges=[(base+lo+i*row,row) for i in selections[match[1]]]
                e['shape'][0]=kept
            n=sum(x[1] for x in ranges)
            e['data_offsets']=[offset,offset+n];offset+=n
            assert name2 not in new and name2 not in index
            new[name2]=e;index[name2]=src.name;spans.append((name,name2,ranges,n))
        blob=json.dumps(new,separators=(',',':')).encode();blob+=b' '*((-len(blob))%8)
        dest=temp/src.name
        with src.open('rb',buffering=0) as inp,dest.open('wb',buffering=0) as out:
            out.write(struct.pack('<Q',len(blob)));out.write(blob)
            for oldname,newname,ranges,n in spans:
                digest=hashlib.sha256()
                for pos,size in ranges:
                    inp.seek(pos)
                    while size:
                        chunk=inp.read(min(size,4<<20))
                        if not chunk:raise IOError('Short checkpoint read')
                        out.write(chunk);digest.update(chunk);size-=len(chunk)
                manifest.append({'source':oldname,'target':newname,'bytes':n,'sha256':digest.hexdigest()})
            out.flush();os.fsync(out.fileno())
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(inp.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(out.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        total+=offset
        print(json.dumps({'event':'shard_exported','file':src.name,'bytes':dest.stat().st_size}),flush=True)
    for src in source.iterdir():
        if not src.is_file() or src.suffix=='.safetensors' or src.name in {'config.json','model.safetensors.index.json'}:continue
        if src.name in {'quantization_config.json','exl3_k_map.json'}:
            (temp/src.name).write_text(json.dumps(metadata(json.loads(src.read_text())),separators=(',',':')))
        else:shutil.copy2(src,temp/src.name)
    tc['n_routed_experts']=kept
    config['pruning_provenance']={'original_experts':original,'retained_experts':kept,
        'method':'packed expert selection; see selection.json and calibration manifest',
        'weight_transform':'unchanged retained packed bytes; router rows sliced and expert IDs remapped'}
    (temp/'config.json').write_text(json.dumps(config,indent=2))
    (temp/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':index},separators=(',',':')))
    (temp/'selection.json').write_text(json.dumps(selections))
    with (temp/'tensor-manifest.jsonl').open('w') as f:
        for item in manifest:f.write(json.dumps(item)+'\n')
    # Read the entire output back and verify bytes, bounds, shapes, names and routing cardinalities.
    expected={x['target']:x for x in manifest}; observed=set()
    for path in sorted(temp.glob('*.safetensors')):
        base,h=header(path)
        with path.open('rb') as f:
            for name,e in h.items():
                if name=='__metadata__':continue
                lo,hi=e['data_offsets']; assert 0<=lo<=hi<=path.stat().st_size-base
                f.seek(base+lo); remaining=hi-lo;digest=hashlib.sha256()
                while remaining:
                    b=f.read(min(remaining,4<<20));assert b;digest.update(b);remaining-=len(b)
                assert digest.hexdigest()==expected[name]['sha256'],name
                assert hi-lo==expected[name]['bytes']; observed.add(name)
    assert observed==set(expected)==set(index)
    for i in range(layers):
        actual={int(EXPERT.match(k)[3]) for k in observed if EXPERT.match(k) and int(EXPERT.match(k)[2])==i}
        assert actual==set(range(kept)),(i,len(actual))
    report={'original_experts':original,'retained_experts':kept,'output_tensor_bytes':total,
            'verified_tensors':len(observed),'wall_seconds':time.monotonic()-started,
            'source':str(source),'destination':str(destination)}
    (temp/'EXPORT_VERIFIED.json').write_text(json.dumps(report,indent=2))
    temp.rename(destination)
    print(json.dumps({'event':'export_verified',**report}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);p.add_argument('--selection',type=pathlib.Path,required=True);a=p.parse_args()
    export(a.source,a.destination,json.loads(a.selection.read_text()))

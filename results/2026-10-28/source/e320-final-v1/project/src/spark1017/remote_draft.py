"""Delta-synchronized DSpark helper; the original target verifies every token.

The rank leader sends only newly accepted projected KV rows. Both target ranks
receive the same proposals, or a common fallback decision, on their control link.
No target weights, target KV, sampling rule, or verification code is changed.
"""
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import struct
import time
import uuid


class PacketConnection(http.client.HTTPConnection):
    """Write bounded binary requests in one call, avoiding split small packets.

    The helper protocol uses byte bodies only. Streaming/chunked bodies retain
    the standard library implementation. This is also tested without a network.
    """
    def _send_output(self, message_body=None, encode_chunked=False):
        if isinstance(message_body, (bytes, bytearray)) and not encode_chunked:
            packet = b'\r\n'.join(self._buffer) + b'\r\n\r\n' + message_body
            self._buffer.clear()
            self.send(packet)
        else:
            super()._send_output(message_body, encode_chunked)


def delta_start(owner, end, previous, ring_size):
    """Use a delta only within the same request and an unbroken ring horizon."""
    return (previous[1] if previous and previous[0] == owner
            and 0 <= end-previous[1] <= ring_size else end-ring_size)


class Client:
    def __init__(self):
        endpoint = os.environ['SPARK1017_DRAFT_HELPER'].rsplit(':', 1)
        self.http = PacketConnection(endpoint[0], int(endpoint[1]), timeout=2.0)
        self.key = Path(os.environ['SPARK1017_DRAFT_KEY']).read_text().strip()
        self.session = uuid.uuid4().hex
        self.synced = {}
        self.calls = self.failures = self.bytes = 0
        self.seconds = 0.0
        self.retry_after = 0.0

    def run(self, runtime, tokens, positions, slots):
        if time.monotonic() < self.retry_after:
            return None
        import torch
        dp = runtime.dpool
        owners = {s.st.index: s.sid for s in runtime.streams.values() if not s.done}
        updates, tensors = [], []
        for slot, end in zip(slots, positions):
            owner = f'{self.session}:{owners[slot]}'
            previous = self.synced.get(slot)
            start = delta_start(owner,end,previous,dp.ring_size)
            # Full snapshots include negative positions at the beginning. They
            # address unused ring rows and are masked by the drafter attention.
            index = (torch.arange(start, end, device='cuda') % dp.ring_size + slot*dp.ring_size)
            tensors.extend(r[index].contiguous().view(-1) for r in dp.rings)
            updates.append({'slot':slot, 'owner':owner, 'start':start, 'end':end})
        payload = torch.cat(tensors).view(torch.uint16).cpu().numpy().tobytes()
        header = json.dumps({'v':1,'tokens':tokens,'positions':positions,'slots':slots,
                             'updates':updates}, separators=(',',':')).encode()
        body = struct.pack('!I',len(header)) + header + payload
        start_time = time.perf_counter()
        try:
            self.http.request('POST','/draft',body,{'Authorization':'Bearer '+self.key,
                              'Content-Type':'application/octet-stream'})
            response = self.http.getresponse()
            answer = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError('helper refused synchronization: '+str(answer.get('error')))
            if (len(answer['drafts']) != len(tokens) or len(answer['conf']) != len(tokens)
                    or any(len(r) != runtime.e.drafter.size for r in answer['drafts'])
                    or any(not isinstance(v,int) or v<0 or v>=runtime.m.cfg.vocab
                           for row in answer['drafts'] for v in row)):
                raise ValueError('helper proposal contract mismatch')
            self.synced.update({u['slot']:(u['owner'],u['end']) for u in updates})
            self.calls += 1; self.bytes += len(body)
            self.seconds += time.perf_counter()-start_time
            if self.calls % 200 == 0:
                print(json.dumps({'event':'remote_draft_progress','calls':self.calls,
                      'failures':self.failures,'sent_bytes':self.bytes,
                      'rpc_mean_ms':1000*self.seconds/self.calls}),flush=True)
            return answer
        except Exception:
            self.http.close(); self.synced.clear(); self.failures += 1
            self.retry_after = time.monotonic() + 30.0
            raise


def propose(runtime, tokens, positions, slots):
    """Return proposals or None. Both ranks choose fallback together."""
    if runtime.e.rank == 0:
        if not hasattr(runtime,'_remote_draft'):
            runtime._remote_draft=Client()
        try:
            answer=runtime._remote_draft.run(runtime,tokens,positions,slots)
            packet=['spark1017-draft',answer]
        except Exception as exc:
            packet=['spark1017-draft',None]
            print('[spark1017] helper unavailable; both ranks use local drafter: '+type(exc).__name__,flush=True)
        if runtime.link is not None: runtime.link.send(packet)
    else:
        packet=runtime._spark1017_follow_link.receive()
        if not packet or packet[0]!='spark1017-draft':
            raise RuntimeError('remote draft rank protocol mismatch')
    answer=packet[1]
    if answer is None:return None
    runtime.drafters[len(tokens)].last_conf=answer['conf']
    return answer['drafts']


def serve(model_dir, key_file, host, port):
    """One authenticated, bounded binary endpoint. Never deserializes code."""
    import torch
    import triton
    import triton.language as tl
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from tensorfold.families.deepseek_v41.cuda.weights import load
    from tensorfold.families.deepseek_v41.cuda.model import Model,Comm
    from tensorfold.families.deepseek_v41.cuda.dspark import Drafter,DraftPool,BatchDraftGraph

    @triton.jit(do_not_specialize=['start','slot','count','offset'])
    def scatter_delta(src, r0, r1, r2, start, slot, count, offset,
                      WIDTH:tl.constexpr, RING:tl.constexpr, BLOCK:tl.constexpr):
        row=tl.program_id(0); col=tl.arange(0,BLOCK)
        target=((start+row+RING)%RING+slot*RING)*WIDTH+col
        source=offset+row*WIDTH+col
        tl.store(r0+target,tl.load(src+source,col<WIDTH,0),col<WIDTH)
        tl.store(r1+target,tl.load(src+source+count*WIDTH,col<WIDTH,0),col<WIDTH)
        tl.store(r2+target,tl.load(src+source+2*count*WIDTH,col<WIDTH,0),col<WIDTH)

    # Test the transport copy independently of model numerics, including all
    # 16-bit payload patterns, negative initial positions, wrap and slot isolation.
    with torch.inference_mode():
        cases=0
        for count in (1,3,16,128,144):
            for start in (-144,-1,0,137,131071):
                for slot in (0,1):
                    source=torch.randint(-32768,32767,(3*count*512,),dtype=torch.int16,device='cuda').view(torch.bfloat16)
                    rings=[torch.zeros((288,512),dtype=torch.bfloat16,device='cuda') for _ in range(3)]
                    expect=[x.clone() for x in rings]
                    idx=torch.arange(start,start+count,device='cuda')%144+slot*144
                    for i in range(3):expect[i][idx]=source.view(3,count,512)[i]
                    scatter_delta[(count,)](source,*rings,start,slot,count,0,512,144,512)
                    assert all(torch.equal(x.view(torch.int16),y.view(torch.int16)) for x,y in zip(rings,expect))
                    cases+=1
        print(json.dumps({'event':'delta_kernel_verified','cases':cases,'bit_exact':True}),flush=True)

    root=Path(model_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    with (root/'draft.safetensors').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256'], 'drafter pack hash mismatch'
    key=Path(key_file).read_text().strip()
    w=load(root,0,1,n_layers=0,dspark=True)
    model=Model(w,Comm(None,1));drafter=Drafter(model);pool=DraftPool(drafter,2)
    seq=model.new_cache(524288); graphs={}
    with torch.inference_mode():
        for n in (1,2):
            graph=BatchDraftGraph(drafter,seq,pool,n)
            graph.tokens.fill_(1000);graph.q0.fill_(128);graph.slots.copy_(torch.arange(n,device='cuda'))
            graph.capture();graphs[n]=graph
        timings=[]
        for _ in range(100):
            t=time.perf_counter();graphs[1].run([1000],[128],[0]);timings.append(time.perf_counter()-t)
    print(json.dumps({'event':'helper_ready','gpu':torch.cuda.get_device_name(),
          'memory_gib':torch.cuda.memory_allocated()/2**30,'median_draft_ms':sorted(timings)[50]*1000}),flush=True)
    states={}

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            import socket
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*args):pass
        def answer(self,status,obj):
            data=json.dumps(obj,separators=(',',':')).encode()
            # One complete bounded response also avoids proxy delayed-ACK stalls.
            header=(f'HTTP/1.1 {status} {self.responses[status][0]}\r\n'
                    f'Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n').encode()
            self.wfile.write(header+data)
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+key):
                self.answer(401,{'error':'authorization'});return
            self.answer(200,{'ready':True})
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+key):
                self.close_connection=True;self.answer(401,{'error':'authorization'});return
            length=int(self.headers.get('Content-Length','0'))
            if not 4 <= length <= 2*1024*1024:
                self.close_connection=True;self.answer(400,{'error':'length'});return
            raw=self.rfile.read(length)
            received=time.perf_counter()
            try:
                if len(raw)!=length:raise ValueError('short payload')
                hlen=struct.unpack('!I',raw[:4])[0]
                if hlen>8192:raise ValueError('header length')
                meta=json.loads(raw[4:4+hlen]);data=memoryview(raw)[4+hlen:]
                tokens,positions,slots=meta['tokens'],meta['positions'],meta['slots']
                n=len(tokens)
                if (meta['v']!=1 or n not in (1,2) or len(positions)!=n or len(slots)!=n
                    or len(set(slots))!=n or any(s not in (0,1) for s in slots)
                    or any(not isinstance(t,int) or not 0<=t<w.cfg.vocab for t in tokens)
                    or any(not isinstance(p,int) or not 0<=p<524288 for p in positions)
                    or len(meta['updates'])!=n):raise ValueError('request shape')
                at=0;prepared=[]
                for u,slot,end in zip(meta['updates'],slots,positions):
                    count=u['end']-u['start']
                    if u['slot']!=slot or u['end']!=end or not 0<=count<=pool.ring_size:
                        raise ValueError('update range')
                    if count!=pool.ring_size and states.get(slot)!=(u['owner'],u['start']):
                        raise ValueError('resync required')
                    size=count*w.cfg.head_dim*2
                    if at+3*size>len(data):raise ValueError('short ring payload')
                    prepared.append((u,at//2));at+=3*size
                if at!=len(data):raise ValueError('trailing payload')
                with torch.inference_mode():
                    packed=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).cuda() if len(data) else None
                    for u,offset in prepared:
                        count=u['end']-u['start']
                        if count:
                            scatter_delta[(count,)](packed,*pool.rings,u['start'],u['slot'],count,offset,
                                    w.cfg.head_dim,pool.ring_size,triton.next_power_of_2(w.cfg.head_dim))
                        states[u['slot']]=(u['owner'],u['end'])
                    begin=time.perf_counter();drafts=graphs[n].run(tokens,positions,slots)
                    answer={'drafts':drafts,'conf':graphs[n].last_conf,'draft_ms':(time.perf_counter()-begin)*1000,
                            'server_ms':(time.perf_counter()-received)*1000}
                self.answer(200,answer)
            except Exception as exc:
                states.clear();self.answer(400,{'error':type(exc).__name__+': '+str(exc)[:120]})
    HTTPServer((host,port),Handler).serve_forever()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--key-file',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=19992)
    a=p.parse_args();serve(a.model,a.key_file,a.host,a.port)

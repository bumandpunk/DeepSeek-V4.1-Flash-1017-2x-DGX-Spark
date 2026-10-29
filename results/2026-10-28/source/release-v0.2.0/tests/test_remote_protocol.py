"""Protocol tests cover packet framing and rank agreement on failure."""
from types import SimpleNamespace
from spark1017.remote_draft import PacketConnection,propose,Client
import time

def test_cooldown_before_gpu_or_network():
    client=Client.__new__(Client)
    client.retry_after=time.monotonic()+30
    assert client.run(None,None,None,None) is None

def test_one_write():
    class Capture(PacketConnection):
        def __init__(self):super().__init__("test");self.writes=[]
        def send(self,data):self.writes.append(data)
    for payload in (b"",b"abc",bytes(442368)):
        c=Capture();c.request("POST","/draft",payload,{"Authorization":"Bearer test"})
        assert len(c.writes)==1
        header,body=c.writes[0].split(b"\r\n\r\n",1)
        assert body==payload
        assert b"Content-Length: "+str(len(payload)).encode() in header

def test_rank_decision():
    class Link:
        def send(self,p):self.packet=p
        def receive(self):return self.packet
    class Client:
        fail=False
        def run(self,*args):
            if self.fail:raise ConnectionError("injected")
            return {"drafts":[[11,12]],"conf":[[.3,.4]]}
    link=Link();client=Client()
    head=SimpleNamespace(e=SimpleNamespace(rank=0),link=link,_remote_draft=client,drafters={1:SimpleNamespace()})
    follower=SimpleNamespace(e=SimpleNamespace(rank=1),link=None,_spark1017_follow_link=link,drafters={1:SimpleNamespace()})
    assert propose(head,[1],[128],[0])==propose(follower,[1],[128],[0])==[[11,12]]
    assert head.drafters[1].last_conf==follower.drafters[1].last_conf
    client.fail=True
    assert propose(head,[1],[128],[0]) is None
    assert propose(follower,[1],[128],[0]) is None

if __name__=="__main__":test_one_write();test_rank_decision();test_cooldown_before_gpu_or_network();print("protocol tests passed")

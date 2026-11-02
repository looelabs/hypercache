"""Does the concurrency of WRITES change how fast the same data reads back? One client per thread, like the adapter."""
import sys,time,zlib,threading,os,mmap
from concurrent.futures import ThreadPoolExecutor as P
from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockReader, DirectBlockWriter
SZ=14*1024*1024; N=256; sock=sys.argv[1]; WP=int(sys.argv[2]); tl=threading.local(); w=DirectBlockWriter(max_workers=8)
buf=mmap.mmap(-1,SZ); buf.write(os.urandom(SZ)); v=memoryview(buf); L=[None]*N
def cl():
    if not hasattr(tl,'c'): tl.c=HyperCacheClient(socket_path=sock)
    return tl.c
def put(i):
    c=cl(); u=f"hypercache://kv_cache/w/{i}.kv"; s=c.stage_write(u,SZ,tier="nvme",auto_flush_s3=False); w.write_chunks(s.chunk_layout,v)
    p=0;cr=[]
    for ch in s.chunk_layout: cr.append(zlib.crc32(v[p:p+ch.length])&0xffffffff); p+=ch.length
    assert c.commit_write(u,local_only=True,chunk_checksums=cr); L[i]=c.locate(u,auto_ingest=False).chunk_layout
t=time.time()
with P(WP) as e: list(e.map(put,range(N)))
off=[l[0].offset for l in L]; seq=sum(1 for a,b in zip(off,off[1:]) if b-a==SZ+ (4194304-SZ%4194304 if SZ%4194304 else 0) or abs(b-a)<=16*1024*1024)
print(f"write_parallel={WP}: {N*SZ/1e6/(time.time()-t):.0f} MB/s",flush=True)
def rd(wk,o):
    r=DirectBlockReader(max_workers=wk,verify_checksums=True); bufs=[memoryview(mmap.mmap(-1,SZ)) for _ in range(o)]
    tb=threading.local(); idx=iter(range(o)); lk=threading.Lock()
    def one(l):
        if not hasattr(tb,'b'):
            with lk: tb.b=bufs[next(idx)]
        r.read_range_into(l,0,SZ,tb.b)
    t=time.time()
    with P(o) as e: list(e.map(one,L))
    d=time.time()-t; r.close(); print(f"  read parallel={o:2} io_workers={wk:2}: {N*SZ/1e6/d:5.0f} MB/s",flush=True)
for o,wk in [(4,4),(8,8),(16,16),(4,4),(8,8),(16,16)]: rd(wk,o)

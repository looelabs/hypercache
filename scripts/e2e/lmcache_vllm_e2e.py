"""Sends a long prompt twice; between sends L1 is flushed by volume, so the second must come from L2."""
import json,sys,time,urllib.request
URL="http://127.0.0.1:18111/v1/completions"; M="Qwen/Qwen2.5-7B-Instruct-AWQ"
def ask(prompt):
    body=json.dumps({"model":M,"prompt":prompt,"max_tokens":24,"temperature":0,"stream":True}).encode()
    req=urllib.request.Request(URL,body,{"Content-Type":"application/json"}); t=time.time(); first=None; text=""
    with urllib.request.urlopen(req,timeout=300) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            c=json.loads(line[5:])["choices"][0]["text"]
            if c and first is None: first=time.time()-t
            text+=c
    return first,text
base=" ".join(f"Item {i} in the inventory has the serial code {i*7919%10007} and is stored on shelf {i%37}." for i in range(380))
out={}
out["cold"]=ask("A: "+base+"\nSummary:")
out["warm"]=ask("A: "+base+"\nSummary:")
for n in range(int(sys.argv[1])):
    ask(f"Filler {n}: "+" ".join(f"Record {n}-{i} has value {i*n%991}." for i in range(800)))
time.sleep(20)  # let L1 eviction get below its watermark, so the load has room
out["after_l1_flush"]=ask("A: "+base+"\nSummary:")
for k,v in out.items(): print(k,f"ttft={v[0]:.3f}s",repr(v[1][:70]))
print("identical_output",out["cold"][1]==out["warm"][1]==out["after_l1_flush"][1])

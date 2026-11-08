"""E0.4: each worker is pinned to one MIG instance by UUID and reports what it sees.
Usage: python scripts/e0_mig_test.py            (reads MIG UUIDs from `nvidia-smi -L`)"""
import os, re, subprocess, sys, json
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    import torch, time
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(50): y = x @ x
    torch.cuda.synchronize()
    pr = torch.cuda.get_device_properties(0)
    print(json.dumps({"worker": sys.argv[2], "pinned_to": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_uuid": str(pr.uuid), "name": pr.name, "mem_gib": round(pr.total_memory / 2**30, 1),
                      "sms": pr.multi_processor_count, "matmul_s": round(time.time() - t, 3)}))
    sys.exit(0)
out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
uuids = re.findall(r"(MIG-[0-9a-f-]+)", out)
print("MIG instances found:", len(uuids))
# one process per instance; the big instance also gets extra processes (see 01 D6)
big = [u for u in uuids if u == uuids[-1]]
jobs = [(i, u) for i, u in enumerate(uuids)] + [(len(uuids) + k, uuids[-1]) for k in range(3)]
procs = [(i, u, subprocess.Popen([sys.executable, __file__, "--worker", str(i)], env={**os.environ, "CUDA_VISIBLE_DEVICES": u},
         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)) for i, u in jobs]
for i, u, p in procs:
    o, e = p.communicate(); print(i, o.strip() or ("FAIL " + e.strip()[-300:]))

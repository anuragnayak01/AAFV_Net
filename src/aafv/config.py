import hashlib, json, yaml

def load(path):
    with open(path) as f:
        return yaml.safe_load(f)

def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

def evidence(name, cfg, **kv):
    print(f"=== EVIDENCE {name} config_hash={config_hash(cfg)}")
    for k, v in kv.items():
        print(f"  {k}: {v}")

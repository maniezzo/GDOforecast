import json, sys

config_path = sys.argv[1]
overrides   = json.loads(sys.argv[2])

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.update(overrides)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
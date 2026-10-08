import csv,json
from pathlib import Path
from statistics import mean,stdev
ROOT=Path("controlled_quantity_outputs");CONFIGS=("balanced","moderate","extreme");METHODS=("fedavg_nnpu","complement_nnpu","fedeppu")
records={}; duplicates={}
for file in ROOT.rglob("*.json"):
    try:
        payload=json.loads(file.read_text(encoding="utf-8"))
        key=(payload["size_config"],payload["method"],int(payload["seed"]))
    except (KeyError,ValueError,json.JSONDecodeError):
        continue
    if key in records:
        duplicates.setdefault(key,[]).append(records[key][0])
        duplicates[key].append(file)
    if key not in records or file.stat().st_mtime > records[key][0].stat().st_mtime:
        records[key]=(file,payload)
expected={(config,method,seed) for config in CONFIGS for method in METHODS for seed in range(42,47)}
missing=sorted(expected-set(records))
if missing:
    print(f"discovered_json={len(records)} expected_json=45")
    for config,method,seed in missing:
        print(f"missing config={config} method={method} seed={seed}")
    raise SystemExit("controlled quantity outputs are incomplete; wait for or rerun the missing training tasks")
if duplicates:
    print(f"warning=duplicate_runs_detected keys={len(duplicates)}; using newest JSON for each key")
rows=[]
for config in CONFIGS:
    for method in METHODS:
        for seed in range(42,47):
            file,j=records[(config,method,seed)];m=j["metrics"]
            rows.append({"config":config,"ratio":"/".join(map(str,j["ratio"])),"method":method,"seed":seed,"counts":"/".join(map(str,j["client_counts"])),"priors":"/".join(f"{x:.6f}" for x in j["priors"]),"weights":"/".join(f"{x:.4f}" for x in j["diagnostics"].get("aggregation_weights",j["diagnostics"].get("effective_weights",[]))),"selected_round":m["selected_round"],"acc":m["acc"],"f1":m["f1"],"bacc":m["bacc"],"auc":m["auc"]})
with open("controlled_quantity_per_seed.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for config in CONFIGS:
    for method in METHODS:
        rr=[r for r in rows if r["config"]==config and r["method"]==method];z={"config":config,"ratio":rr[0]["ratio"],"method":method,"counts":rr[0]["counts"],"priors":rr[0]["priors"],"weights":rr[0]["weights"]}
        for key in ("acc","f1","bacc","auc"):
            x=[100*r[key] for r in rr];z[key]=f"{mean(x):.2f} ± {stdev(x):.2f}"
        summary.append(z)
with open("controlled_quantity_summary.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print("completed_runs=45\nsummary=controlled_quantity_summary.csv\nper_seed=controlled_quantity_per_seed.csv")

import csv,json
from pathlib import Path
from statistics import mean,stdev
ROOT=Path("controlled_prior_outputs");CONFIGS=("mild","moderate","extreme");METHODS=("fedavg_nnpu","ranking_fedavg","fedeppu")
rows=[]
for config in CONFIGS:
    for method in METHODS:
        for seed in range(42,47):
            files=list((ROOT/config/method/f"seed{seed}").glob("*.json"))
            if len(files)!=1:raise SystemExit(f"expected_one_json config={config} method={method} seed={seed} found={len(files)}")
            j=json.loads(files[0].read_text(encoding="utf-8"));m=j["metrics"]
            rows.append({"config":config,"method":method,"seed":seed,"priors":"/".join(map(str,j["priors"])),"client_size":j["p_per_client"]+j["u_per_client"],"selected_round":m["selected_round"],"acc":m["acc"],"f1":m["f1"],"bacc":m["bacc"],"auc":m["auc"]})
with open("controlled_prior_per_seed.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for config in CONFIGS:
    for method in METHODS:
        rr=[r for r in rows if r["config"]==config and r["method"]==method];z={"config":config,"method":method,"priors":rr[0]["priors"],"client_size":rr[0]["client_size"]}
        for key in ("acc","f1","bacc","auc"):
            x=[100*r[key] for r in rr];z[key]=f"{mean(x):.2f} ± {stdev(x):.2f}"
        summary.append(z)
with open("controlled_prior_summary.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print("completed_runs=45\nsummary=controlled_prior_summary.csv\nper_seed=controlled_prior_per_seed.csv")

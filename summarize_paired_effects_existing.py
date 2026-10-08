"""Create paired-seed effect estimates from completed support ablations."""
import csv,json,math
from pathlib import Path
from statistics import mean,stdev

ROOT=Path("support_ablation_outputs")
VARIANTS=("no_support","local_only","global_only","local_global")
COMPARISONS=(("local_only","no_support"),("global_only","no_support"),
             ("local_global","global_only"),("local_global","local_only"),
             ("local_global","no_support"))
records={}
for file in ROOT.rglob("*.json"):
    try:
        j=json.loads(file.read_text(encoding="utf-8"));variant=j["metadata"]["variant"];seed=int(j["metadata"]["args"]["seed"])
        if variant in VARIANTS and 42<=seed<=46:records[(variant,seed)]=j["history"]["selected"]
    except (KeyError,ValueError,json.JSONDecodeError):pass
missing=[(v,s) for v in VARIANTS for s in range(42,47) if (v,s) not in records]
if missing:raise SystemExit("missing_runs="+",".join(f"{v}/seed{s}" for v,s in missing))
rows=[]
for treatment,reference in COMPARISONS:
    for metric in ("acc","f1","bacc","auc"):
        d=[100*(float(records[(treatment,s)][metric])-float(records[(reference,s)][metric])) for s in range(42,47)]
        effect=mean(d);sd=stdev(d);half=2.776*sd/math.sqrt(5)
        rows.append({"comparison":f"{treatment} - {reference}","metric":metric,
            **{f"seed{s}":d[s-42] for s in range(42,47)},"mean_effect_pp":effect,
            "sample_sd_pp":sd,"ci95_low_pp":effect-half,"ci95_high_pp":effect+half,
            "positive_seeds":sum(x>0 for x in d),"negative_seeds":sum(x<0 for x in d),"zero_seeds":sum(x==0 for x in d)})
with open("support_ablation_paired_effects.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print("completed_pairs=20\noutput=support_ablation_paired_effects.csv\ninterval=t_95_percent_df4_descriptive")

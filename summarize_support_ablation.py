import csv,json,math
from pathlib import Path
from statistics import mean,stdev
ROOT=Path("support_ablation_outputs");VARIANTS=("no_support","local_only","global_only","local_global")
rows=[];objects={}
for variant in VARIANTS:
    for seed in range(42,47):
        files=list((ROOT/variant/f"seed{seed}").glob("*.json"))
        if len(files)!=1:raise SystemExit(f"expected_one_json variant={variant} seed={seed} found={len(files)}")
        j=json.loads(files[0].read_text(encoding="utf-8"));objects[(variant,seed)]=j;h=j["history"];m=h["selected"];late=h["auc"][-20:]
        rows.append({"variant":variant,"seed":seed,"selected_round":m["round"],"acc":m["acc"],"f1":m["f1"],"bacc":m["bacc"],"auc":m["auc"],"late_auc":mean(late),"late_auc_sd":stdev(late),"peak_to_final":max(h["auc"])-h["auc"][-1]})
with open("support_ablation_per_seed.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for variant in VARIANTS:
    rr=[r for r in rows if r["variant"]==variant];z={"variant":variant}
    for key in ("acc","f1","bacc","auc","late_auc","late_auc_sd","peak_to_final"):
        x=[100*r[key] for r in rr];z[key]=f"{mean(x):.2f} ± {stdev(x):.2f}"
    summary.append(z)
with open("support_ablation_summary.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
diagnostics=[]
for variant in VARIANTS:
    selected=[]
    for seed in range(42,47):
        j=objects[(variant,seed)];priors=j["metadata"]["client_priors"]
        for round_diag in j["history"]["client_diag"]:
            for client,diag in enumerate(round_diag):
                if float(diag["expected_negative_count"]) < 1.0:
                    selected.append((seed,client,priors[client],diag))
    diagnostics.append({"variant":variant,"records":len(selected),
        "effective_ranking_grad_norm":mean(float(x[3]["effective_ranking_grad_norm_mean"]) for x in selected),
        "joint_nonfinite_fraction":mean(float(x[3]["joint_grad_nonfinite_fraction"]) for x in selected),
        "clip_active_fraction":mean(float(x[3]["clip_active_fraction"]) for x in selected),
        "base_weight":mean(float(x[3]["base_aggregation_weight"]) for x in selected),
        "effective_weight":mean(float(x[3]["effective_aggregation_weight"]) for x in selected)})
with open("support_ablation_diagnostics.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=diagnostics[0]);w.writeheader();w.writerows(diagnostics)
comparisons=(("local_only","no_support"),("global_only","no_support"),
             ("local_global","global_only"),("local_global","local_only"),
             ("local_global","no_support"))
paired=[]
for treatment,reference in comparisons:
    for metric in ("acc","f1","bacc","auc"):
        differences=[]
        for seed in range(42,47):
            treated=next(r for r in rows if r["variant"]==treatment and r["seed"]==seed)
            baseline=next(r for r in rows if r["variant"]==reference and r["seed"]==seed)
            differences.append(100.0*(treated[metric]-baseline[metric]))
        effect=mean(differences);sd=stdev(differences);half_width=2.776*sd/math.sqrt(5)
        paired.append({"comparison":f"{treatment} - {reference}","metric":metric,
            "seed42":differences[0],"seed43":differences[1],"seed44":differences[2],
            "seed45":differences[3],"seed46":differences[4],"mean_effect_pp":effect,
            "sample_sd_pp":sd,"ci95_low_pp":effect-half_width,"ci95_high_pp":effect+half_width,
            "positive_seeds":sum(x>0 for x in differences),"negative_seeds":sum(x<0 for x in differences),
            "zero_seeds":sum(x==0 for x in differences)})
with open("support_ablation_paired_effects.csv","w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)
# Under strict determinism, variants with identical active operations must have identical curves.
for seed in (42,46):
    for left,right in (("no_support","local_only"),("global_only","local_global")):
        a=objects[(left,seed)]["history"];b=objects[(right,seed)]["history"]
        for key in ("acc","auc","f1","bacc","threshold"):
            if a[key]!=b[key]:raise SystemExit(f"strict_equivalence_failed seed={seed} pair={left}/{right} metric={key}")
print("completed_runs=20\nstrict_equivalence=passed\nsummary=support_ablation_summary.csv\nper_seed=support_ablation_per_seed.csv\ndiagnostics=support_ablation_diagnostics.csv\npaired_effects=support_ablation_paired_effects.csv")

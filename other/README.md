# Baseline implementations

The supported comparison entry point is `python -m other.run_baselines`.
Every baseline uses its final-round model and a fixed zero-logit decision
threshold. Baselines never inspect the proxy set; proxy threshold calibration
and proxy checkpoint selection remain exclusive to FedEPPU.

The primary comparison runner exposes only the following baselines:

- `fedavg_nnpu`: standard FedAvg with the standard binary nnPU risk;
- `fedprox_nnpu`: standard FedProx with the same nnPU risk;
- `scaffold_nnpu`: SCAFFOLD Option II with the same nnPU risk;
- `fedpu_binary`: the supplied FedPU `MPULoss_V2` adapted to one binary
  positive class and one binary negative class.
- `mrfl_nnpu`: a disclosed binary PU adaptation of MRFL. It uses standard
  nnPU in every local optimizer step, a 20-round warm-up, normalized P/U loss
  components with a two-component GMM for client division, a second GMM for
  reliable latent-positive recovery, and reliability-mass aggregation.
- `centralized_pn`: pooled training with complete PN ground truth, used only
  as the fully supervised upper reference;
- `centralized_nnpu`: pooled PU training with the standard nnPU risk;
- `local_nnpu`: three independent nnPU clients, reported through a fixed
  sample-count-weighted logit ensemble, with no communication.

`data_protocol.py` reproduces the formal PU split and retains a separate
ground-truth label vector. Only `centralized_pn` reads that vector. All PU
methods receive only observed P/U labels and disclosed client priors.

These baselines do not use EB prior correction, prior consistency, a ranking
head, deviation-aware learning rates, reliability weights, bounded
aggregation, model EMA, proxy checkpoint selection, proxy threshold
calibration, early stopping, rollback, or an adaptive compute budget.

`MRFL.py` is exposed as `mrfl_nnpu`, but must be reported as **MRFL-nnPU
adaptation** because the original paper assumes fully labelled multi-class
data and cross-entropy losses. `hetero_select.py` remains a historical
adaptation and is not exposed by the comparison runner.

All methods share the generated client split and backbone only to make the
comparison paired. Sharing data and model capacity is an evaluation control,
not a FedEPPU algorithm component.

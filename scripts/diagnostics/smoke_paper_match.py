import argparse
import importlib.util
import inspect
from pathlib import Path

_bf = Path(__file__).resolve()
for _ancestor in _bf.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _bm.setup(str(_bf))
        break
else:
    raise RuntimeError(
        "bootstrap_path.py not found; keep scripts inside the BrepMFR_PyG repository."
    )

args = argparse.Namespace(
    num_classes=25, dropout=0.3, attention_dropout=0.3, act_dropout=0.3,
    d_model=512, dim_node=256, n_heads=32, n_layers_encode=8,
    batch_size=64, max_epochs=200,
    pre_train=r"C:\Users\D58\Desktop\BrepMFR\results\BrepMFR\0416\031255\best.ckpt",
)

from models.transfer_model import DomainAdapt
m = DomainAdapt(args)

# 1) GRL config
grl = m.domain_adv.grl
print("grl.max_iters =", grl.max_iters, "(expected 1000)")
print("grl.auto_step =", grl.auto_step, "(expected True)")
print("grl.alpha     =", grl.alpha, "(expected 1.0)")
assert grl.max_iters == 1000
assert grl.auto_step is True

# 2) Layerdrop should NOT be forced to 0
enc = m.brep_encoder
print("brep_encoder.layerdrop =", getattr(enc, "layerdrop", "n/a"))

# 3) optimizer: attention must NOT appear in any param group
opt_cfg = m.configure_optimizers()
opt = opt_cfg["optimizer"]
attn_params = set(id(p) for p in m.attention.parameters())
opt_param_ids = set()
for g in opt.param_groups:
    for p in g["params"]:
        opt_param_ids.add(id(p))
overlap = attn_params & opt_param_ids
print("attention params in optimizer?", len(overlap), "(expected 0)")
assert len(overlap) == 0, "attention should be frozen"

# 4) betas must be (0.99, 0.999)
for i, g in enumerate(opt.param_groups):
    lr = g["lr"]
    betas = g["betas"]
    print("param_group[{}] lr={} betas={}".format(i, lr, betas))
    assert betas == (0.99, 0.999)

# 5) scheduler
sched_cfg = opt_cfg["lr_scheduler"]
print("scheduler type :", type(sched_cfg["scheduler"]).__name__)
print("scheduler monitor:", sched_cfg["monitor"])
assert type(sched_cfg["scheduler"]).__name__ == "ReduceLROnPlateau"
assert sched_cfg["monitor"] == "eval_loss"

# 6) optimizer_step override should NOT exist
has_override = "optimizer_step" in DomainAdapt.__dict__
print("has optimizer_step override?", has_override, "(expected False)")
assert not has_override

# 7) dann.py: chunk(2) restored
from models.modules.domain_adv.dann import DomainAdversarialLoss
src = inspect.getsource(DomainAdversarialLoss.forward)
assert "d.chunk(2" in src
print("dann.py uses d.chunk(2, dim=0): OK")

print("SMOKE OK - paper-match revert verified")

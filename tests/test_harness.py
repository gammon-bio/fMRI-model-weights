"""Phase 2 harness tests (docs/phase2_harness_spec.md §9.1), with --fast tiny training.

Two episodes are prepared once per module (fix / arm B / confound_on and read_shortcut / arm A /
confound_off); tests that submit work on copies, since an episode takes exactly one submission.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmritest.baselines.leace import LeaceEraser
from fmritest.harness import commands as cmd
from fmritest.harness.cli import main
from fmritest.harness.comparators import BudgetAccount, head_edit, leace_probe_head
from fmritest.harness.episode import Episode, HarnessRefusal, prepare
from fmritest.harness.keys import FORBIDDEN, episode_seed, load_or_create_key
from fmritest.harness.ledger import Ledger
from fmritest.harness.modelio import (
    HEAD_IN,
    SetCache,
    apply_edit,
    first_stage,
    numpy_weights,
    oracle_readout,
    set_logits,
)
from fmritest.harness.reference import window_logit
from fmritest.models.train import to_windows

BUDGET_B = 600


@pytest.fixture(scope="module")
def key(tmp_path_factory):
    return tmp_path_factory.mktemp("key") / "key"


@pytest.fixture(scope="module")
def ep_b(tmp_path_factory, key):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("epB") / "ep"
    return prepare(root, "test-fix-B", "fix", "B", BUDGET_B, fast=True, key_path=key, log=lambda *a: None).root


@pytest.fixture(scope="module")
def ep_a(tmp_path_factory, key):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("epA") / "ep"
    return prepare(root, "test-read-A", "read_shortcut", "A", 0, condition="confound_off", fast=True,
                   key_path=key, log=lambda *a: None).root


def _copy(root, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(root, dst)
    return dst


def _ws(root, name):
    return root / "workspace" / name


def _predict(root, artifact, tmp_path=None, point=0.6, name="pred.json", task="fix"):
    preds = [{"metric": "test_balanced_auc", "direction": "increase", "point": point,
              "interval90": [0.0, 1.0]}] if task == "fix" else \
        [{"metric": "answer_correct", "direction": "no_change", "point": 0.5, "interval90": [0.0, 1.0]}]
    p = _ws(root, name)
    p.write_text(json.dumps({"hypothesis": "scripted", "artifact": str(artifact), "predictions": preds}))
    return cmd.predict(root, p)


# ---------------------------------------------------------------------------------- (g) seeds
def test_seeds_never_collide_and_key_is_private(tmp_path):
    k = load_or_create_key(tmp_path / "sub" / "key")
    assert len(k) == 32 and (tmp_path / "sub" / "key").stat().st_mode & 0o777 == 0o600
    assert load_or_create_key(tmp_path / "sub" / "key") == k            # reused, not regenerated
    seeds = [episode_seed(k, f"ep{i}") for i in range(3000)]
    assert all(10_000 <= s < 2**31 for s in seeds)
    assert not set(seeds) & (set(range(5)) | set(range(100, 104)) | FORBIDDEN)
    assert len(set(seeds)) == len(seeds)
    assert episode_seed(k, "ep7") == seeds[7]                            # deterministic
    assert episode_seed(b"x" * 32, "ep7") != seeds[7]                    # depends on the key


# ------------------------------------------------------------------------------- (d) ledger
def test_ledger_tamper_detected(tmp_path):
    (tmp_path / "w").mkdir()                                             # the workspace exists before any write
    lg = Ledger(tmp_path / "s" / "log.jsonl", tmp_path / "w" / "mirror.jsonl", tmp_path / "s" / "head.json")
    for i in range(4):
        lg.append("x", {"i": i})
    assert lg.verify() == (True, [])
    good = lg.path.read_text()
    lines = good.splitlines()
    e = json.loads(lines[1])
    e["payload"]["i"] = 99
    lg.path.write_text("\n".join([lines[0], json.dumps(e)] + lines[2:]) + "\n")
    ok, prob = lg.verify()
    assert not ok and any("hash mismatch" in p for p in prob)
    # a consistently rewritten chain is still caught by the mirror and the stored head
    from fmritest.harness.ledger import GENESIS, _canon, entry_hash
    prev, out = GENESIS, []
    for line in lines:
        e = json.loads(line)
        e["payload"]["i"] = -1
        e["prev_hash"] = prev
        e["hash"] = entry_hash(e)
        prev = e["hash"]
        out.append(_canon(e))
    lg.path.write_text("\n".join(out) + "\n")
    ok, prob = lg.verify()
    assert not ok and any("head" in p for p in prob) and lg.warnings()
    lg.path.write_text("\n".join(lines[:-1]) + "\n")                     # truncation
    assert not lg.verify()[0]
    lg.path.write_text(good)
    assert lg.verify()[0] and not lg.warnings()
    assert lg.mirror.stat().st_mode & 0o777 == 0o444                     # mirror is read-only
    lg.path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(RuntimeError):                                    # appending to a bad sealed log is refused
        lg.append("x", {})
    lg.path.write_text(good)


def test_mirror_divergence_is_a_logged_warning_not_a_failure(tmp_path):
    (tmp_path / "w").mkdir()                                             # the workspace exists before any write
    lg = Ledger(tmp_path / "s" / "log.jsonl", tmp_path / "w" / "mirror.jsonl", tmp_path / "s" / "head.json")
    lg.append("x", {"i": 0})
    lg.mirror.chmod(0o644)
    lg.mirror.write_text("agent scribble\n")                             # agent edits the mirror
    assert lg.verify()[0] and lg.warnings()
    lg.append("x", {"i": 1})                                             # does not brick the episode
    types = [e["type"] for e in lg.entries()]
    assert types == ["x", "warning", "x"]
    assert lg.mirror_ok() and lg.verify()[0]
    lg.mirror.chmod(0o644)
    lg.mirror.unlink()                                                   # agent deletes it
    lg.append("x", {"i": 2})
    assert lg.mirror_ok() and [e["type"] for e in lg.entries()][-2:] == ["warning", "x"]


def test_episode_log_verify_cli(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    assert main(["verify", "--root", str(root)]) == 0
    p = root / "sealed" / "log.jsonl"
    txt = p.read_text()
    assert f'"budget":{BUDGET_B}' in txt
    p.write_text(txt.replace(f'"budget":{BUDGET_B}', '"budget":99999'))
    assert main(["verify", "--root", str(root)]) == 1


# ---------------------------------------------------------------------- (f) workspace content
def _scan_workspace(root, key):
    ep = Episode(root)
    secret_strings = [str(ep.meta["seed"]), load_or_create_key(key).hex()]
    sd = ep.model("vanilla").state_dict()
    files = sorted(p for p in ep.ws.rglob("*") if p.is_file())
    allowed_top = {"ARCH.md", "TASK.md", "subjects.json", "log_mirror.jsonl"}
    for p in files:
        rel = p.relative_to(ep.ws)
        assert str(rel) in allowed_top or rel.parts[0] in ("weights", "oracle"), rel
        if p.suffix == ".npz":
            with np.load(p) as z:
                assert set(z.files) == set(sd), rel                    # only model tensors
                for k in z.files:
                    assert z[k].shape == tuple(sd[k].shape)
                    assert ep.cfg["data"]["n_timepoints"] not in z[k].shape
        else:
            txt = p.read_text()
            for s in secret_strings:
                assert s not in txt, (rel, "secret leaked")
    if (ep.ws / "subjects.json").exists():
        sub = json.loads((ep.ws / "subjects.json").read_text())
        assert all(isinstance(i, int) for v in sub.values() for i in v)
    return files


def test_workspace_has_no_data_labels_or_seed(ep_b, ep_a, key):
    fb = _scan_workspace(ep_b, key)
    assert any(p.name == "model.npz" for p in fb)
    fa = {p.name for p in _scan_workspace(ep_a, key)}
    assert {"model_A.npz", "model_B.npz", "ARCH.md", "TASK.md"} <= fa and "subjects.json" not in fa
    ma = Episode(ep_a).meta
    assert ma["fast"] and ma["fast_settings"]["train_steps"] > 0          # fast is recorded
    assert ma["read_shortcut"]["answer"] == "neither"                    # confound_off false-positive arm
    assert ma["config"]["data"]["confound"]["strength"] == 0.0


# --------------------------------------------------------------------------- (c) oracle
def test_oracle_budget_hooks_and_arm_a(ep_b, ep_a, tmp_path):
    root = _copy(ep_b, tmp_path)
    pool = json.loads((root / "workspace" / "subjects.json").read_text())["pool"]
    for h in ("hook_embed", "blocks.0.hook_resid_pre"):
        with pytest.raises(HarnessRefusal, match="not served"):
            cmd.forward(root, "0,1", h)
    with pytest.raises(HarnessRefusal, match="unknown hook"):
        cmd.forward(root, "0,1", "nope")
    with pytest.raises(HarnessRefusal, match="at most"):
        cmd.forward(root, ",".join(map(str, pool[:65])), "")
    with pytest.raises(HarnessRefusal, match="not available"):
        cmd.forward(root, str(max(pool) + 1), "")
    assert cmd.status(root)["used"] == 0                                 # failed calls cost nothing
    r = cmd.forward(root, ",".join(map(str, pool[:10])), "blocks.1.hook_resid_post,head_in,logit")
    assert r["cost"] == 10 * 3 and r["used"] == 30
    with np.load(r["output"]) as z:
        assert z["blocks.1.hook_resid_post"].shape == (10, 30, 64) and z["head_in"].shape == (10, 64)
        assert set(np.unique(z["site"])) <= {0, 1, 2} and z["logit"].shape == (10,)
    big = ",".join(map(str, pool[:64]))
    for _ in range(2):
        cmd.forward(root, big, "head_in,blocks.0.hook_resid_post")        # 64 x 3 = 192 units each
    st = cmd.status(root)
    assert st["used"] == 30 + 2 * 192 and st["remaining"] == BUDGET_B - 414
    with pytest.raises(HarnessRefusal, match="remain"):
        cmd.forward(root, big, "head_in,blocks.0.hook_resid_post")        # 414 + 192 > 600
    assert cmd.status(root)["used"] == 414
    assert main(["forward", "--root", str(root), "--subjects", "0", "--hooks", "hook_embed"]) == 2
    types = [e["type"] for e in Episode(root).ledger.entries()]
    assert types.count("forward") == 3 and types.count("refused") >= 6
    with pytest.raises(HarnessRefusal, match="arm A"):
        cmd.forward(_copy(ep_a, tmp_path / "a"), "0", "")


def test_oracle_weights_override(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    W = dict(np.load(root / "workspace" / "weights" / "model.npz"))
    e = _ws(root, "e.npz")
    np.savez(e, **{"head.bias": W["head.bias"] + 5.0})
    a = cmd.forward(root, "0,1,2", "")
    b = cmd.forward(root, "0,1,2", "", weights=str(e))
    with np.load(a["output"]) as za, np.load(b["output"]) as zb:
        np.testing.assert_allclose(zb["logit"], za["logit"] + 5.0, atol=1e-5)
    np.savez(e, **{"head.bias": np.zeros((2,), np.float32)})
    with pytest.raises(HarnessRefusal, match="shape"):
        cmd.forward(root, "0", "", weights=str(e))


# ------------------------------------------------------------- equivalences the scorer relies on
def test_leace_probe_folds_into_head():
    rng = np.random.default_rng(0)
    n, d = 300, 16
    site = rng.integers(0, 3, n)
    y = (rng.random(n) < np.where(site == 1, 0.2, 0.8)).astype(int)
    E = rng.standard_normal((n, d)) + 1.5 * np.eye(d)[site][:, :d] + 0.7 * y[:, None] * np.eye(d)[5]
    Et = rng.standard_normal((50, d))
    for erase in (True, False):
        w, b, parts = leace_probe_head(E, site, y, 0.1, erase=erase)
        er = LeaceEraser().fit(E, site) if erase else None
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        clf.fit(er.transform(E) if erase else E, y)
        ref = clf.decision_function(er.transform(Et) if erase else Et)
        np.testing.assert_allclose(Et @ w + b, ref, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Et @ w + b, parts["clf"].decision_function(
            parts["eraser"].transform(Et) if erase else Et), rtol=1e-9, atol=1e-9)


def test_head_edit_matches_probe_on_model(ep_b):
    ep = Episode(ep_b)
    m = ep.model("vanilla")
    d, s = ep.data(), ep.splits()["splits"]
    tr, tb = s["train"], s["test_balanced"]
    E = oracle_readout(m, d["X"][tr], [HEAD_IN])[HEAD_IN]
    w, b, parts = leace_probe_head(E, d["site"][tr], d["diagnosis"][tr], 0.1)
    me = apply_edit(m, head_edit(w, b))
    got = oracle_readout(me, d["X"][tb], [HEAD_IN])
    ref = parts["clf"].decision_function(parts["eraser"].transform(got[HEAD_IN].astype(np.float64)))
    np.testing.assert_allclose(got["logit"], ref, atol=2e-4)


def test_arch_md_equations_match_model(ep_b):
    ep = Episode(ep_b)
    m = ep.model("vanilla")
    W = numpy_weights(m)
    X = ep.data()["X"][:2]
    xw = to_windows(X, m.window).astype(np.float32)[:, :2].reshape(-1, X.shape[2], m.window)
    with torch.no_grad():
        t = m(torch.from_numpy(xw)).numpy()
    ref = np.array([window_logit(W, x, m.blocks[0].attn.h, len(m.blocks))[0] for x in xw])
    np.testing.assert_allclose(ref, t, atol=1e-4)
    assert "blocks.1.mlp.W_out.weight" in (ep.ws / "ARCH.md").read_text()


def test_staged_set_logits_are_bitwise_identical(ep_b):
    ep = Episode(ep_b)
    m = ep.model("vanilla")
    X = ep.data()["X"][ep.splits()["splits"]["test_balanced"][:20]]
    c = SetCache(m, X)
    full = set_logits(m, c, 0)
    for st in range(1, len(m.blocks) + 3):
        np.testing.assert_array_equal(set_logits(m, c, st), full)
    W = numpy_weights(m)["blocks.1.mlp.W_out.weight"] * 0.5
    edit = {"blocks.1.mlp.W_out.weight": W}
    me = apply_edit(m, edit)
    assert first_stage(edit, 2) == 2
    np.testing.assert_array_equal(set_logits(me, c, 2), set_logits(me, c, 0))


# ------------------------------------------------- (a) scripted LEACE agent reproduces comparator
def test_scripted_leace_agent_reproduces_comparator(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    ep = Episode(root)
    s = ep.splits()
    g2a = {g: i for i, g in enumerate(s["agent_to_global"])}
    ids = [g2a[g] for g in s["splits"]["train"].tolist()]               # privileged: which ids are train
    E, site, y = [], [], []
    for i in range(0, len(ids), 64):
        r = cmd.forward(root, ids[i:i + 64], HEAD_IN)
        with np.load(r["output"]) as z:
            E.append(z[HEAD_IN])
            site.append(z["site"])
            y.append(z["diagnosis"])
    assert cmd.status(root)["used"] == 2 * len(ids) <= BUDGET_B
    w, b, _ = leace_probe_head(np.concatenate(E), np.concatenate(site), np.concatenate(y), 0.1)
    edit = _ws(root, "edit.npz")
    np.savez(edit, **head_edit(w, b))
    with np.load(edit) as z1, np.load(root / "sealed" / "comparator_edits" / "posthoc_leace_site.npz") as z2:
        for k in z2.files:
            np.testing.assert_array_equal(z1[k], z2[k])
    pid = _predict(root, edit, tmp_path)
    cmd.submit(root, pid, edit=str(edit))
    res = cmd.score(root)
    comp = json.loads((root / "sealed" / "comparators.json").read_text())["posthoc_leace_site"]
    assert abs(res["metrics"]["test_balanced"]["bacc"] - comp["metrics"]["test_balanced"]["bacc"]) <= 1e-9
    assert abs(res["metrics"]["test_site"]["auc"] - comp["metrics"]["test_site"]["auc"]) <= 1e-9
    assert all(res["restore_check"].values())
    assert res["diff"]["tensors_changed"] == ["head.bias", "head.weight"] and res["diff"]["total_rank"] == 2
    assert res["nulls"]["matched"]["outcomes"]["test_balanced_bacc"]["n"] == ep.n_null()
    assert res["primary_metric"] == "test_balanced_auc" and res["primary"]["percentile_matched_null"] is not None
    for m in ("test_balanced_auc", "test_balanced_bacc", "test_balanced_bacc_pool_median_threshold",
              "test_site_auc", "test_site_bacc", "test_site_bacc_pool_median_threshold"):
        for kind in ("matched", "variance"):
            assert 0 <= res["nulls"][kind]["outcomes"][m]["percentile"] <= 100
        assert m in res["vs_comparators"]["unedited"]["agent_minus_comparator"]
    assert res["nulls"]["directions"]["site_energy"].startswith("lower")
    assert res["metrics"]["network_readout"]["backbone_changed"] is False
    assert res["log_verified"] and main(["verify", "--root", str(root)]) == 0


# --------------------------------------------------------- (b) random matched edit lies in the null
def test_scripted_random_edit_inside_null(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    W = dict(np.load(root / "workspace" / "weights" / "model.npz"))
    rng = np.random.default_rng(2024)
    A = W["blocks.1.mlp.W_out.weight"].astype(np.float64)
    u, v = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[1])
    D = np.outer(u, v)
    D *= 0.5 * np.linalg.norm(A) / np.linalg.norm(D)
    edit = _ws(root, "rand.npz")
    np.savez(edit, **{"blocks.1.mlp.W_out.weight": (A + D).astype(np.float32)})
    pid = _predict(root, edit, tmp_path)
    cmd.submit(root, pid, edit=str(edit))
    res = cmd.score(root)
    assert res["diff"]["per_tensor"]["blocks.1.mlp.W_out.weight"]["rank"] == 1
    for m in ("test_balanced_bacc", "test_balanced_auc"):
        pct = res["nulls"]["matched"]["outcomes"][m]["percentile"]
        assert 2.5 < pct < 97.5, (m, pct)
    assert res["nulls"]["variance"]["outcomes"]["test_balanced_bacc"]["n"] == Episode(root).n_null()
    assert res["metrics"]["network_readout"]["backbone_changed"] is True
    mech = res["nulls"]["matched"]["mechanism_max_cos"]["blocks.1.mlp.W_out.weight"]["resid"]
    assert 0 <= mech["percentile"] <= 100


# ------------------------------------------------------------------------- (e) submit refusals
def test_submit_requires_matching_prediction_and_only_once(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    W = dict(np.load(root / "workspace" / "weights" / "model.npz"))
    e1, e2 = _ws(root, "e1.npz"), _ws(root, "e2.npz")
    np.savez(e1, **{"head.bias": W["head.bias"] + 0.1})
    np.savez(e2, **{"head.bias": W["head.bias"] - 0.1})
    with pytest.raises(HarnessRefusal, match="no prediction"):
        cmd.submit(root, "P0", edit=str(e1))                             # P0 is the prepare entry
    with pytest.raises(HarnessRefusal, match="no prediction"):
        cmd.submit(root, "P999", edit=str(e1))
    pid = _predict(root, e1, tmp_path)
    with pytest.raises(HarnessRefusal, match="sha256"):
        cmd.submit(root, pid, edit=str(e2))                              # wrong file for this prediction
    bad = _ws(root, "bad.npz")
    np.savez(bad, **{"head.weight": np.full((1, 64), np.nan, np.float32)})
    pbad = _predict(root, bad, tmp_path, name="pbad.json")
    with pytest.raises(HarnessRefusal, match="non-finite"):
        cmd.submit(root, pbad, edit=str(bad))
    with pytest.raises(HarnessRefusal, match="test_balanced_auc"):
        p = _ws(root, "nopred.json")
        p.write_text(json.dumps({"hypothesis": "h", "artifact": str(e1), "predictions": []}))
        cmd.predict(root, p)
    r = cmd.submit(root, pid, edit=str(e1))
    assert r["diff"]["tensors_changed"] == ["head.bias"] and r["diff"]["n_weights_changed"] == 1
    p2 = _predict(root, e2, tmp_path, name="p2.json")
    with pytest.raises(HarnessRefusal, match="already"):
        cmd.submit(root, p2, edit=str(e2))
    with pytest.raises(HarnessRefusal, match="already"):
        cmd.submit(root, pid, edit=str(e1))
    assert Episode(root).ledger.verify()[0]


def test_read_shortcut_answer_scored(ep_a, tmp_path):
    root = _copy(ep_a, tmp_path)
    ans = _ws(root, "answer.json")
    ans.write_text(json.dumps({"answer": "neither", "where": "nowhere", "confidence": 0.6}))
    pid = _predict(root, ans, tmp_path, task="read")
    cmd.submit(root, pid, answer=str(ans))
    res = cmd.score(root)
    assert res["correct"] is True and res["answer"]["where"] == "nowhere"


# -------------------------------------------------------------------------- (h) restore / hash
def test_score_detects_modified_checkpoint(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    W = dict(np.load(root / "workspace" / "weights" / "model.npz"))
    e = _ws(root, "e.npz")
    np.savez(e, **{"head.bias": W["head.bias"] + 0.1})
    cmd.submit(root, _predict(root, e, tmp_path), edit=str(e))
    ck = root / "sealed" / "ckpt" / "vanilla.pt"
    sd = torch.load(ck, weights_only=True)
    sd["head.bias"] += 1.0
    torch.save(sd, ck)
    with pytest.raises(RuntimeError, match="changed on disk"):
        cmd.score(root)
    assert main(["verify", "--root", str(root)]) == 1


def test_budget_account_units():
    a = BudgetAccount(10)
    assert a.cost(3, 2) == 9 and a.can(9) and not a.can(11)
    a.charge(9)
    with pytest.raises(RuntimeError):
        a.charge(2)


# ------------------------------------------------------------- repair round 1 (Fable review)
def test_no_served_hook_is_affine_in_the_raw_input(ep_b):
    """Generic leak test: every served hook must not be an affine function of the raw input window
    (nor the input an affine function of it); the two known exact leaks must be caught by the check."""
    from fmritest.harness.leakcheck import leak_report
    from fmritest.harness.modelio import hook_names, served_hooks
    ep = Episode(ep_b)
    m = ep.model("vanilla")
    thr = ep.hcfg["oracle"]["leak_r2_threshold"]
    tr = ep.splits()["splits"]["train"]
    rep = leak_report(m, ep.data()["X"][tr], hook_names(m) + [HEAD_IN], thr, seed=1, fast=True)
    assert rep["forward_r2"]["hook_embed"] > thr and rep["forward_r2"]["blocks.0.hook_resid_pre"] > thr
    refused = ep.refused_hooks()
    assert {"hook_embed", "blocks.0.hook_resid_pre"} <= refused
    assert set(rep["flagged"]) <= refused
    for h in served_hooks(m, refused):
        assert rep["forward_r2"][h] <= thr, (h, rep["forward_r2"][h])
        assert rep["inverse_r2"].get(h, 0.0) <= thr, (h, rep["inverse_r2"][h])
    assert set(ep.meta["leak_check"]["flagged"]) <= refused                 # stored at prepare
    assert "blocks.0.hook_resid_pre" not in (ep.ws / "TASK.md").read_text().split("Refused")[0]


def test_paths_confined_to_workspace(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    sealed_edit = root / "sealed" / "comparator_edits" / "posthoc_leace_site_pool.npz"
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.forward(root, "0,1", "", weights=str(sealed_edit))               # the reviewer's attack
    link = _ws(root, "innocent.npz")
    link.symlink_to(sealed_edit)
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.forward(root, "0,1", "", weights=str(link))                      # symlink out of the workspace
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.forward(root, "0,1", "", weights="../sealed/comparator_edits/posthoc_leace_site_pool.npz")
    outside = tmp_path / "ids.json"
    outside.write_text("[0, 1]")
    with pytest.raises(HarnessRefusal, match="@file"):
        cmd.forward(root, str(outside), "")                                  # a path without @ is not a file
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.forward(root, "@" + str(outside), "")
    assert cmd.status(root)["used"] == 0                                     # refusals cost nothing
    _ws(root, "ids.json").write_text("[0, 1]")
    W = dict(np.load(_ws(root, "weights/model.npz")))
    np.savez(_ws(root, "mine.npz"), **{"head.bias": W["head.bias"] + 1.0})
    r = cmd.forward(root, "@ids.json", "", weights="mine.npz")              # relative to the workspace
    assert r["cost"] == 2
    ents = Episode(root).ledger.entries()
    fw = [e for e in ents if e["type"] == "forward"][-1]["payload"]
    assert fw["weights"] == "mine.npz" and fw["subjects_arg"] == "@ids.json"
    reasons = [e["payload"]["reason"] for e in ents if e["type"] == "refused"]
    assert len(reasons) == 5 and all("workspace" in x or "@file" in x for x in reasons)
    # predict: spec and artifact must be inside the workspace; refusals are logged
    spec = tmp_path / "pred.json"
    spec.write_text(json.dumps({"hypothesis": "h", "artifact": str(_ws(root, "mine.npz")), "predictions": [
        {"metric": "test_balanced_auc", "direction": "increase", "point": 0.6, "interval90": [0.5, 0.7]}]}))
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.predict(root, spec)
    _ws(root, "p_out.json").write_text(json.dumps({"hypothesis": "h", "artifact": str(sealed_edit), "predictions": [
        {"metric": "test_balanced_auc", "direction": "increase", "point": 0.6, "interval90": [0.5, 0.7]}]}))
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.predict(root, _ws(root, "p_out.json"))
    assert [e["type"] for e in Episode(root).ledger.entries()][-2:] == ["refused", "refused"]
    # submit: a prediction bound to the hash of an outside file cannot be submitted from outside
    _ws(root, "p_sha.json").write_text(json.dumps({"hypothesis": "h", "artifact_sha256": cmd.file_sha256(sealed_edit),
                                                   "predictions": [{"metric": "test_balanced_auc", "direction":
                                                                    "increase", "point": 0.6,
                                                                    "interval90": [0.5, 0.7]}]}))
    pid = cmd.predict(root, _ws(root, "p_sha.json"))
    with pytest.raises(HarnessRefusal, match="inside the workspace"):
        cmd.submit(root, pid, edit=str(sealed_edit))
    assert not (root / "sealed" / "submission.json").exists()


def test_budget_lock_under_concurrent_calls(ep_b, tmp_path):
    import subprocess
    import sys
    root = _copy(ep_b, tmp_path)
    ids = "0-63"                                                             # arm B: every agent id is in the pool
    hooks = "head_in,blocks.0.hook_resid_post,blocks.1.hook_resid_post,blocks.1.hook_resid_mid"
    cost = 64 * 5                                                            # two calls: 640 > 600
    env = {"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}
    import fmritest
    env["PYTHONPATH"] = str(Path(fmritest.__file__).resolve().parents[1])
    procs = [subprocess.Popen([sys.executable, "-m", "fmritest.harness", "forward", "--subjects", ids,
                               "--hooks", hooks], env=env, cwd=root / "workspace", stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) for _ in range(2)]
    rcs = sorted(p.wait(timeout=180) for p in procs)
    st = cmd.status(root)
    assert st["used"] <= st["budget"]
    assert rcs == [0, 2] and st["used"] == cost and st["n_calls"] == 1
    assert Episode(root).ledger.verify()[0]


def test_task_md_has_no_root_and_root_is_inferred(ep_b, monkeypatch, capsys):
    txt = (ep_b / "workspace" / "TASK.md").read_text()
    assert str(ep_b) not in txt and "sealed/" not in txt and "/sealed" not in txt and "--root" not in txt
    assert "inside this" in txt                                              # paths stated relative to the workspace
    monkeypatch.chdir(ep_b / "workspace")
    capsys.readouterr()
    assert main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["budget"] == BUDGET_B
    e = Episode(ep_b)
    assert e.meta["isolation"] == e.hcfg["isolation_label"]                 # label recorded from the config


def test_pool_only_scripted_leace_matches_info_matched_comparator(ep_b, tmp_path):
    """Arm-B agent without privileged knowledge: queries the whole pool in subjects.json order."""
    root = _copy(ep_b, tmp_path)
    pool = json.loads(_ws(root, "subjects.json").read_text())["pool"]
    E, site, y = [], [], []
    for i in range(0, len(pool), 64):
        r = cmd.forward(root, pool[i:i + 64], HEAD_IN)
        with np.load(r["output"]) as z:
            E.append(z[HEAD_IN])
            site.append(z["site"])
            y.append(z["diagnosis"])
    assert cmd.status(root)["used"] == 2 * len(pool) <= BUDGET_B
    w, b, _ = leace_probe_head(np.concatenate(E), np.concatenate(site), np.concatenate(y), 0.1)
    edit = _ws(root, "edit.npz")
    np.savez(edit, **head_edit(w, b))
    with np.load(edit) as z1, np.load(root / "sealed" / "comparator_edits" / "posthoc_leace_site_pool.npz") as z2:
        for k in z2.files:
            np.testing.assert_array_equal(z1[k], z2[k])
    cmd.submit(root, _predict(root, edit), edit="edit.npz")
    res = cmd.score(root)
    comps = json.loads((root / "sealed" / "comparators.json").read_text())
    comp = comps["posthoc_leace_site_pool"]
    assert comp["primary_for_arm"] == ["B"] and comps["unedited"]["primary_for_arm"] == ["A"]
    for k in ("bacc", "auc", "bacc_pool_median_threshold"):
        assert abs(res["metrics"]["test_balanced"][k] - comp["metrics"]["test_balanced"][k]) <= 1e-9
    # bias-invariant outcomes and reader site energy are reported
    assert 0 <= res["metrics"]["test_balanced"]["bacc_pool_median_threshold"] <= 1
    mech = res["nulls"]["mechanism"]["head.weight"]
    assert mech["tensor_kind"] == "reader" and set(mech["site_energy"]) == {"old", "new", "change"}
    assert res["nulls"]["matched"]["site_energy_new"]["head.weight"]["n"] == Episode(root).n_null()
    assert res["nulls"]["mechanism"]["head.bias"]["tensor_kind"] == "other"
    assert all(v == 0.0 for v in res["vs_comparators"]["posthoc_leace_site_pool"]["agent_minus_comparator"].values())
    assert res["vs_comparators"]["posthoc_leace_site_pool"]["primary_for_arm"] == ["B"]


def test_pool_median_threshold_is_bias_invariant(ep_b):
    ep = Episode(ep_b)
    sc = ep.scorer()
    W = numpy_weights(sc.model)
    base = sc.metrics({})
    shifted = sc.metrics({"head.bias": W["head.bias"] + 3.0})
    for k in ("test_balanced", "test", "test_site"):
        assert shifted[k]["bacc_pool_median_threshold"] == base[k]["bacc_pool_median_threshold"]
        assert shifted[k]["auc"] == pytest.approx(base[k]["auc"])
    assert shifted["pool_median_threshold"] == pytest.approx(base["pool_median_threshold"] + 3.0, abs=1e-5)


# ------------------------------------------------------------- repair round 2 (reviewer re-check)
_HARNESS_WRITTEN = {"log.jsonl", "log_head.json", "budget.json", "log.lock", "budget.lock"}


def _sealed_hashes(root):
    out = {}
    for p in sorted((root / "sealed").rglob("*")):
        rel = p.relative_to(root / "sealed")
        if p.is_file() and rel.parts[0] != ".tmp" and p.name not in _HARNESS_WRITTEN:
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _assert_episode_usable(root):
    ep = Episode(root)
    st = json.loads((root / "sealed" / "budget.json").read_text())
    assert set(st) == {"budget", "used", "n_calls"}                          # not overwritten by the mirror
    ep.model("vanilla")                                                      # checkpoint hash still verifies
    r = cmd.forward(root, "5,6", "")
    assert Path(r["output"]).is_file() and not Path(r["output"]).is_symlink()
    assert ep.ledger.verify()[0] and ep.ledger.mirror_ok()
    assert main(["verify", "--root", str(root)]) == 0


def test_symlinked_log_mirror_does_not_write_through(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    before = _sealed_hashes(root)
    budget_before = (root / "sealed" / "budget.json").read_bytes()
    m = root / "workspace" / "log_mirror.jsonl"
    m.chmod(0o644)
    m.unlink()
    m.symlink_to(root / "sealed" / "budget.json")                           # the reviewer's attack
    W = dict(np.load(_ws(root, "weights/model.npz")))
    np.savez(_ws(root, "e.npz"), **{"head.bias": W["head.bias"]})
    _predict(root, _ws(root, "e.npz"))                                       # appends -> writes the mirror
    assert (root / "sealed" / "budget.json").read_bytes() == budget_before
    assert _sealed_hashes(root) == before
    assert not m.is_symlink() and Episode(root).ledger.mirror_ok()
    ents = Episode(root).ledger.entries()
    assert any(e["type"] == "warning" and "symlink" in e["payload"]["message"] for e in ents)
    _assert_episode_usable(root)


@pytest.mark.parametrize("attack", ["symlink_file", "hardlink_file", "symlink_dir"])
def test_planted_links_in_oracle_dir_do_not_write_through(ep_b, tmp_path, attack):
    root = _copy(ep_b, tmp_path)
    ck = root / "sealed" / "ckpt" / "vanilla.pt"
    odir = root / "workspace" / "oracle"
    if attack == "symlink_file":                                             # the reviewer's attack
        for k in range(2):
            (odir / f"call_{k:04d}.npz").symlink_to(ck)
    elif attack == "hardlink_file":
        for k in range(2):
            os.link(ck, odir / f"call_{k:04d}.npz")
    else:
        odir.rmdir()
        odir.symlink_to(root / "sealed" / "ckpt", target_is_directory=True)
    before = _sealed_hashes(root)
    r = cmd.forward(root, "0,1", "head_in")
    with np.load(r["output"]) as z:
        assert z["head_in"].shape == (2, 64)
    assert _sealed_hashes(root) == before                                    # checkpoint byte-identical
    assert not odir.is_symlink() and not Path(r["output"]).is_symlink() and os.stat(r["output"]).st_nlink == 1
    msgs = [e["payload"]["message"] for e in Episode(root).ledger.entries() if e["type"] == "warning"]
    assert msgs and any(w in msgs[0] for w in ("symlink", "hard link"))
    _assert_episode_usable(root)
    assert _sealed_hashes(root) == before


def test_agent_cannot_address_another_episode(ep_b, tmp_path, monkeypatch, capsys):
    X, Y = _copy(ep_b, tmp_path / "x"), _copy(ep_b, tmp_path / "y")
    y_log, y_budget = (Y / "sealed" / "log.jsonl").read_bytes(), (Y / "sealed" / "budget.json").read_bytes()
    monkeypatch.chdir(X / "workspace")
    for argv in (["forward", "--root", str(Y), "--subjects", "0,1", "--hooks", "head_in"],
                 ["status", "--root", str(Y)], ["predict", "--root", str(Y), "--spec", "x.json"]):
        assert main(argv) == 2
    assert (Y / "sealed" / "log.jsonl").read_bytes() == y_log and (Y / "sealed" / "budget.json").read_bytes() == y_budget
    refused = [e for e in Episode(X).ledger.entries() if e["type"] == "refused"]
    assert refused and "is not the episode" in refused[-1]["payload"]["reason"]
    assert main(["forward", "--root", str(X), "--subjects", "0,1", "--hooks", ""]) == 0     # own root is fine
    assert main(["forward", "--subjects", "0", "--hooks", ""]) == 0                          # inferred
    sub = X / "workspace" / "oracle"
    monkeypatch.chdir(sub)                                                   # inside the workspace also works
    assert main(["status"]) == 0
    monkeypatch.chdir(tmp_path)                                              # outside any workspace: refused
    assert main(["status", "--root", str(X)]) == 2
    assert main(["verify"]) == 2


# -------------------------------------------------------------------------- pre-registration conventions
def test_no_submission_scored_as_unedited(ep_b, tmp_path):
    root = _copy(ep_b, tmp_path)
    with pytest.raises(cmd.HarnessRefusal):
        cmd.score(root)
    res = cmd.score(root, as_unedited=True)
    comps = json.loads((root / "sealed" / "comparators.json").read_text())
    assert res["no_submission"] is True
    assert res["primary"]["value"] == comps["unedited"]["metrics"]["test_balanced"]["auc"]
    assert res["primary"]["percentile_matched_null"] == 50.0
    assert res["vs_comparators"]["unedited"]["agent_minus_comparator"]["test_balanced_auc"] == 0.0
    assert cmd.verify(root)["ok"]


def test_no_submission_read_is_incorrect(ep_a, tmp_path):
    root = _copy(ep_a, tmp_path)
    res = cmd.score(root, as_unedited=True)
    assert res["correct"] is False and res["no_submission"] is True


def test_exclusion_flag_in_ledger_before_agent(ep_b):
    from fmritest.harness.episode import Episode
    ep = Episode(ep_b)
    first = [e for e in ep.ledger.entries() if e["type"] == "prepare"][0]
    assert first["payload"]["exclusion"]["fix_include"] == ep.meta["exclusion"]["fix_include"]

# EXP-25 id-0 hotfix: two backend-integration bugs surfaced by the anchor probe.
import sys

# ---- Bug B: orthonormalize() device mismatch (powersgd_activation.py:127) ----
# The degenerate-column repair builds q_fix on CPU (deterministic CPU generator)
# then torch.where(bad, q_fix, q) where q is on cuda -> cross-device RuntimeError.
p1 = "verl/workers/comm_eff/powersgd_activation.py"
s = open(p1).read()
old1 = '''        q_fix, _ = torch.linalg.qr(rand, mode="reduced")
        q = torch.where(bad.unsqueeze(0), q_fix, q)'''
new1 = '''        q_fix, _ = torch.linalg.qr(rand, mode="reduced")
        # q lives on the compute device (cuda); q_fix was built on CPU (the
        # deterministic CPU generator). Move the repair frame onto q's device +
        # dtype before torch.where so the degenerate-column repair does not raise
        # a cross-device error (EXP-25 id-0 hotfix).
        q_fix = q_fix.to(device=q.device, dtype=q.dtype)
        q = torch.where(bad.unsqueeze(0), q_fix, q)'''
assert s.count(old1) == 1, "Bug B: expected exactly 1 match, got %d" % s.count(old1)
s = s.replace(old1, new1)
open(p1, "w").write(s)
print("Bug B fixed: q_fix -> q.device in orthonormalize (%s)" % p1)

# ---- Bug A: anchor staleness assert off-by-one at the warmup boundary ----
# (transformer_impl.py ~1263). Steps are 1-BASED (no step 0). At step == delay_K
# the requested snapshot is step - delay_K == 0, which NEVER existed, so the
# t-delay_K snapshot is genuinely unavailable until step == delay_K + 1. The
# guard `if step >= delay_K` fires the hard-assert one step too early and crashes
# (also bites production delay_K=5 on step 5). Correct post-warmup boundary is
# step > delay_K (i.e. step >= delay_K + 1), when step - delay_K >= 1 is a step
# that actually ran.
p2 = "verl/workers/engine/fsdp/transformer_impl.py"
t = open(p2).read()
old2 = '''        if int(step) >= int(delay_K):
            assert _used_step == _req_step, (
                f"comm_eff anchor staleness: post-warmup step={step} requested the t-delay_K "'''
new2 = '''        # 1-based steps (no step 0): the t-delay_K snapshot only becomes available
        # at step == delay_K + 1 (at step == delay_K the request is step 0, which
        # never existed). So the post-warmup guarantee holds for step > delay_K,
        # NOT step >= delay_K — the latter hard-asserts one step too early and
        # crashes the FIRST eligible step (delay_K=1 -> step 1; delay_K=5 -> step
        # 5). (EXP-25 id-0 hotfix.)
        if int(step) > int(delay_K):
            assert _used_step == _req_step, (
                f"comm_eff anchor staleness: post-warmup step={step} requested the t-delay_K "'''
assert t.count(old2) == 1, "Bug A: expected exactly 1 match, got %d" % t.count(old2)
t = t.replace(old2, new2)
open(p2, "w").write(t)
print("Bug A fixed: staleness assert boundary >= -> > delay_K (%s)" % p2)

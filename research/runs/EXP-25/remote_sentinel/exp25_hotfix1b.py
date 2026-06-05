# Refine Bug B fix to device-ONLY move (preserve CPU QR for cross-rank bit-identity;
# do not touch dtype). Matches coordinator guidance exactly.
p1 = "verl/workers/comm_eff/powersgd_activation.py"
s = open(p1).read()
old = '''        q_fix, _ = torch.linalg.qr(rand, mode="reduced")
        # q lives on the compute device (cuda); q_fix was built on CPU (the
        # deterministic CPU generator). Move the repair frame onto q's device +
        # dtype before torch.where so the degenerate-column repair does not raise
        # a cross-device error (EXP-25 id-0 hotfix).
        q_fix = q_fix.to(device=q.device, dtype=q.dtype)
        q = torch.where(bad.unsqueeze(0), q_fix, q)'''
new = '''        q_fix, _ = torch.linalg.qr(rand, mode="reduced")
        # q lives on the compute device (cuda); q_fix was built on CPU via the
        # DETERMINISTIC cpu generator + a CPU QR (keep it on CPU so the repair is
        # bit-identical across ranks — a GPU QR could differ in low bits and
        # break the cross-rank Q consensus). Move ONLY the result onto q's device
        # right before torch.where so the degenerate-column repair does not raise
        # a cross-device error. dtype is already fp32 on both (EXP-25 id-0 hotfix).
        q = torch.where(bad.unsqueeze(0), q_fix.to(q.device), q)'''
assert s.count(old) == 1, "refine Bug B: expected exactly 1 match, got %d" % s.count(old)
s = s.replace(old, new)
open(p1, "w").write(s)
print("Bug B refined: device-only move (CPU QR preserved) in orthonormalize")

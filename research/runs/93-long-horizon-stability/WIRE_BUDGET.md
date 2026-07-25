# Wire budget table for every round-A arm, computed from source before a5 launches

Prompted by the a3 verdict's finding that `comm_eff/logical_pp_bytes_prf` is a **coordinate count despite the `_bytes_` name**, and that a5's FRLR payload lands in that same field. Fixing this now, because a5's E2 eligibility verdict (wire <= 1232 bits/token/boundary) depends on reading it correctly, and a naive read is wrong by 16x.

## The two accounting paths in source

**PRF / FRLR path** (`state.py:465-483`) writes a COORDINATE COUNT into `comm_eff/logical_pp_bytes_prf`:
- PRF: `(1.0 - p) * hidden_size`
- FRLR: `frlr_payload_per_token`, defined at `activation_mask.py:555` as `kept = r_eff + k + (0 if frlr_unbiased else 1)`

**sr_quant path** (`activation_quant.py:490,498`) writes actual BITS into `comm_eff/logical_pp_bits_sr_quant`:
- full mode: `hidden_size * bits + n_blocks * 16`
- subset mode: `subset_k * bits + subset_k * 16 / eff_block`

So the two families are reported in **different units**. Coordinates convert to bits at **16 bits per coordinate** (fp16 on the wire).

## Framework validated against two known values before use

- a3 subset mode: `493*2 + 493*16/32` = 986 + 246.5 = **1232.5 bits**, matching the a3 verdict exactly.
- a1 full mode: `1536*1 + 48*16` = 1536 + 768 = **2304 bits**, matching the 1.8701x figure exactly.
- incumbent: the a3 verdict's deletion term `H/k - 1 = 18.948` implies `k = 1536/19.948 = 77` kept coordinates, and `77 * 16 = 1232 bits`, which is exactly the registered incumbent budget. Three independent consistency checks pass.

## The table

| arm | codec | payload | bits/token/boundary | vs incumbent 1232 | E2 (<= 1232) |
|---|---|---|---|---|---|
| incumbent `90-prf-exactk-600` | PRF exact-k p=0.95 | 77 coords | **1232** | 1.0000x | pass |
| a1 | sr_quant 1-bit, all coords | 1536 vals + 48 scales | **2304** | **1.8701x** | **FAIL** |
| a3 | sr_quant 2-bit on 493-coord subset | 493 vals + scales | **1232.5** | 1.0004x | **FAIL by 0.5 bits** |
| a4 | PRF exact-k p=0.95 **plus CVC** | 77 coords | **1232** | 1.0000x | **pass** |
| a5 | FRLR r48 k28 **plus token-IS** | 48 + 28 + 1 = **77 coords** | **1232** | 1.0000x | **pass** |

## Consequences

1. **a5 sits at exact parity with the incumbent, 1232 bits, and passes E2.** It was designed to: `rank + k + 1 = 48 + 28 + 1 = 77` is precisely the incumbent's kept-coordinate count.
2. **a4 is also at exact parity**, because a4 IS the incumbent's codec and CVC is a loss term with zero wire cost. Likewise token-IS on a5 is a training-side reweighting and costs no bits.
3. **The reporting hazard is live for a5**: its field will read **77**, not 1232. Comparing that raw number against a3's 1232.5 or a1's 2304 would understate a5's cost by 16x and would wrongly make it look like the cheapest arm by an order of magnitude. **Multiply the PRF/FRLR field by 16 before any comparison.**
4. Only a1 has a real budget problem (1.87x). a3's 0.5-bit overage is 0.04 percent and `subset_k=492` would give exactly 1230.0.

So on wire budget the ranking is: a4 = a5 = incumbent (parity) < a3 (+0.04 percent) << a1 (+87 percent). Two of the three arms that could plausibly win round A are at exact parity, which means a communication-efficiency claim is available to either of them without any "accounting of the trade" caveat.

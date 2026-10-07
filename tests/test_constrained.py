#!/usr/bin/env python3
"""Offline test for the v3 constrained-decoding scorer logic (no vLLM/GPU)."""
import math
from raim.backends import (_choice_token_ids, _score_from_constrained_logprobs,
                           _allowed_token_ids, SCORER_REV)

assert SCORER_REV == "v3_constrained", SCORER_REV

class LP:
    def __init__(self, logprob): self.logprob = logprob

# gemma-like ids from the real diagnostic: Yes=3553 No=1294 ▁Yes=6287 ▁No=1307 yes=3276 no=956
pos_ids = {3553, 6287, 3276, 9999, 8888, 7777}   # 6, as observed
neg_ids = {1294, 1307, 956, 6666, 5555, 4444, 3333}  # 7, as observed

def score(lp):  # calls the actual production function, not a copy of it
    return _score_from_constrained_logprobs(lp, pos_ids, neg_ids)

# 1. mass on the SPACE variant only (the v1/v2-fatal case): still scored, 100% coverage
lp = {6287: LP(math.log(0.7)), 1307: LP(math.log(0.3))}   # ▁Yes vs ▁No
c, p = score(lp); assert c == 1 and abs(p - 0.7) < 1e-9, (c, p)

# 2. renormalisation over the constrained set (masked logits excluded)
lp = {3553: LP(math.log(0.18)), 1294: LP(math.log(0.02))}  # sums to 0.2 pre-norm
c, p = score(lp); assert c == 1 and abs(p - 0.9) < 1e-9, (c, p)

# 3. No wins
lp = {3553: LP(math.log(0.3)), 1294: LP(math.log(0.7))}
c, p = score(lp); assert c == 0 and abs(p - 0.3) < 1e-9, (c, p)

# 4. mass split across BOTH pos variants is summed (sum-over-bucket, not max)
lp = {3553: LP(math.log(0.3)), 6287: LP(math.log(0.3)), 1294: LP(math.log(0.4))}
c, p = score(lp); assert c == 1 and abs(p - 0.6) < 1e-9, (c, p)

# 5. the production candidate set is the union, deduplicated, sorted
allowed = _allowed_token_ids(pos_ids, neg_ids)
assert allowed == sorted(set(allowed)) and set(allowed) == pos_ids | neg_ids
assert _allowed_token_ids({3, 1}, {1, 2}) == [1, 2, 3]   # overlap counted once
print(f"all constrained-decoding tests passed (SCORER_REV={SCORER_REV}, |allowed|={len(allowed)})")

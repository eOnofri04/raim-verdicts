#!/usr/bin/env python3
"""Offline tests for the v2 score_choice matcher (no vLLM/GPU needed)."""
from raim.backends import (_choice_token_ids, _score_v1_decoded_string,
                           _score_from_logprobs, SCORER_REV)

class FakeTok:
    def __init__(self, vocab): self._v = vocab
    def get_vocab(self): return self._v

class LP:  # mimics vllm Logprob
    def __init__(self, logprob, decoded): self.logprob, self.decoded_token = logprob, decoded

POS, NEG = ("Yes", "yes"), ("No", "no")
posset = {w.strip().lower() for w in POS}
negset = {w.strip().lower() for w in NEG}

# --- 1. id-set construction across tokenizer families -----------------------
sp = FakeTok({"\u2581Yes": 1, "\u2581No": 2, "Yes": 3, "no": 4, "\u2581eyes": 5,
              "\u2581Not": 6, "YES": 7, "\u2581yes": 8, "Noah": 9, "\u2581NO": 10})
bpe = FakeTok({"\u0120Yes": 11, "\u0120No": 12, "Yes": 13, "No": 14,
               "\u010aYes": 15, "Yes,": 16, "\u0120YES": 17, "yes": 18,
               "\u0120Nob": 19})
assert _choice_token_ids(sp, POS) == {1, 3, 7, 8}, _choice_token_ids(sp, POS)
assert _choice_token_ids(sp, NEG) == {2, 4, 10}, _choice_token_ids(sp, NEG)
assert _choice_token_ids(bpe, POS) == {11, 13, 15, 17, 18}
assert _choice_token_ids(bpe, NEG) == {12, 14}
print("1. id-set construction (SP metaspace / byte-BPE / exclusions)  OK")

# --- 2. the v1 failure mode now resolves -------------------------------------
pos_ids, neg_ids = _choice_token_ids(sp, POS), _choice_token_ids(sp, NEG)
# gemma/Yi scenario: top-k holds only marker-prefixed pieces; v1 saw None.
lp_gemma = {1: LP(-0.3, "\u2581Yes"), 2: LP(-1.5, "\u2581No"), 99: LP(-2.0, "the")}
v1_pos = any((i.decoded_token or "").strip().lower() in posset for i in lp_gemma.values())
v1_neg = any((i.decoded_token or "").strip().lower() in negset for i in lp_gemma.values())
assert not v1_pos and not v1_neg, "scenario should be invisible to the v1 matcher"
choice, p = _score_from_logprobs(lp_gemma, pos_ids, neg_ids, posset, negset)
assert choice == 1 and 0.7 < p < 0.8, (choice, p)
print(f"2. v1-dead scenario resolves under v2: choice={choice} p_yes={p:.3f}     OK")

# --- 3. one-sided saturation now two-sided -----------------------------------
lp_onesided = {3: LP(-0.9, "Yes"), 2: LP(-0.4, "\u2581No")}   # v1 saw only 'Yes' -> (1, 1.0)
choice, p = _score_from_logprobs(lp_onesided, pos_ids, neg_ids, posset, negset)
assert choice == 0 and p < 0.5, (choice, p)
print(f"3. gemma-it saturation case now two-sided: choice={choice} p={p:.3f}    OK")

# --- 4. clean-model parity: fallback string path still fires -----------------
lp_clean = {300: LP(-0.2, " Yes"), 301: LP(-2.2, " No")}      # ids in NEITHER set
choice, p = _score_from_logprobs(lp_clean, pos_ids, neg_ids, posset, negset)
assert choice == 1 and p > 0.85
lp_empty = {99: LP(-0.1, "the"), 98: LP(-0.2, "I")}
assert _score_from_logprobs(lp_empty, pos_ids, neg_ids, posset, negset) == (None, None)
print("4. v1 fallback parity + (None,None) on a true miss               OK")

# --- 5. superset property: any v1 match is a v2 match ------------------------
import itertools, random
rng = random.Random(0)
pieces = list(sp.get_vocab().items()) + [(" Yes", 300), (" no", 301), ("the", 99)]
for trial in range(200):
    lp = {tid: LP(-rng.random() * 5, piece) for piece, tid in rng.sample(pieces, 4)}
    c1, _ = _score_v1_decoded_string(lp, posset, negset)
    c2, _ = _score_from_logprobs(lp, pos_ids, neg_ids, posset, negset)
    if c1 is not None:
        assert c2 is not None, "v2 must cover every v1 match"
print("5. 200 random scenarios: v2 matches are a superset of v1's       OK")
print(f"\nall matcher tests passed (SCORER_REV={SCORER_REV})")

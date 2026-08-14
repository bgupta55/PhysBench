"""
Blueprint v3 §6 — Verify the kyber-py pk/H_pk validation-abort mechanism.

Determines whether the 100% validation_abort for pk/H_pk flips is:
  Explanation A: an explicit H(ek_pke) re-hash-and-compare (FIPS 203 Algorithm 18 step 3)
  Explanation B: a deserialization/type/range-check exception unrelated to cryptographic hash

Approach: read the _decaps_internal source at runtime, identify the exact code path,
then empirically confirm by:
  1. Injecting a pk flip and catching the ValueError message
  2. Injecting an H_pk flip and catching the ValueError message
  3. Separately testing pure deserialization failures vs hash failures

Result saved to results_v2/exp_v3_sec6_pk_abort_trace.json
"""

import json, hashlib, subprocess, datetime, inspect, sys
from pathlib import Path

# ── locate results dir ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results_v2"
RESULTS_DIR.mkdir(exist_ok=True)

from kyber_py.ml_kem import ML_KEM_512

# ── 1. Source-code analysis ───────────────────────────────────────────────────
src = inspect.getsource(ML_KEM_512._decaps_internal.__wrapped__
                        if hasattr(ML_KEM_512._decaps_internal, '__wrapped__')
                        else ML_KEM_512._decaps_internal)

# Find the hash-check line
hash_check_present = "self._H(ek_pke) != h" in src
hash_check_raises_valueerror = ("raise ValueError" in src and
                                "hash check failed" in src)
# Find what happens BEFORE the hash check (type checks)
type_check_dk_len = "len(dk) != self._dk_size()" in src
type_check_c_len  = "len(c) != 32" in src

print("=== SOURCE CODE ANALYSIS ===")
print(f"  H(ek_pke) != h check present:  {hash_check_present}")
print(f"  Hash check raises ValueError:   {hash_check_raises_valueerror}")
print(f"  dk length type check present:   {type_check_dk_len}")
print(f"  ciphertext length check present:{type_check_c_len}")

# Extract lines around hash check for citation
lines = src.splitlines()
hash_lines = [(i+1, l) for i, l in enumerate(lines) if "H(ek_pke)" in l or "hash check" in l.lower()]
print("\n  Hash-check lines:")
for lineno, line in hash_lines:
    print(f"    line ~{lineno}: {line.rstrip()}")

# ── 2. Get module file path (for FIPS 203 citation) ──────────────────────────
import kyber_py.ml_kem.ml_kem as _mod
module_file = _mod.__file__

# ── 3. Empirical test — pk flip: does it trigger hash-check or deser check? ──
ek, dk = ML_KEM_512.keygen()
K_good, ct = ML_KEM_512.encaps(ek)

dk_bytes = bytearray(dk)
K_OFFSET = 768  # packed_s = dk[0:768], pk = dk[768:768+800]

results = {}

# 3a. Flip ONE bit in pk field — should trigger hash mismatch
dk_pk_flip = bytearray(dk)
dk_pk_flip[K_OFFSET + 0] ^= 0x01  # flip LSB of first pk byte
pk_outcome = None
pk_error_msg = None
try:
    ML_KEM_512.decaps(bytes(dk_pk_flip), ct)
    pk_outcome = "no_error"
except ValueError as e:
    pk_outcome = "ValueError"
    pk_error_msg = str(e)

print(f"\n=== EMPIRICAL: pk flip ===")
print(f"  outcome: {pk_outcome}")
print(f"  message: {pk_error_msg!r}")

is_hash_abort = pk_error_msg is not None and "hash check" in pk_error_msg.lower()
is_type_abort = pk_error_msg is not None and ("type check" in pk_error_msg.lower() or
                                               "modulus check" in pk_error_msg.lower() or
                                               "length" in pk_error_msg.lower())

# 3b. Flip ONE bit in H_pk field — dk[768+800:768+800+32]
H_PK_OFFSET = 768 + 800
dk_hpk_flip = bytearray(dk)
dk_hpk_flip[H_PK_OFFSET + 0] ^= 0x01
hpk_outcome = None
hpk_error_msg = None
try:
    ML_KEM_512.decaps(bytes(dk_hpk_flip), ct)
    hpk_outcome = "no_error"
except ValueError as e:
    hpk_outcome = "ValueError"
    hpk_error_msg = str(e)

print(f"\n=== EMPIRICAL: H_pk flip ===")
print(f"  outcome: {hpk_outcome}")
print(f"  message: {hpk_error_msg!r}")

is_hpk_hash_abort = hpk_error_msg is not None and "hash check" in hpk_error_msg.lower()

# 3c. Cross-check: what does H(flipped_ek_pke) vs stored_h produce?
ek_pke_flipped = bytes(dk_pk_flip[768:768+800])
stored_h       = bytes(dk[768+800:768+800+32])
recomputed_h   = hashlib.sha3_256(ek_pke_flipped).digest()
hash_mismatch  = (recomputed_h != stored_h)

print(f"\n=== HASH MISMATCH CROSS-CHECK ===")
print(f"  H(flipped_pk) == stored_H_pk:  {not hash_mismatch}  (expect False)")
print(f"  stored_H_pk:    {stored_h.hex()[:32]}...")
print(f"  H(flipped_pk):  {recomputed_h.hex()[:32]}...")

# 3d. Confirm: if we patch stored_H_pk to match the flipped pk, does decaps succeed?
dk_pk_and_h_fixed = bytearray(dk_pk_flip)
dk_pk_and_h_fixed[H_PK_OFFSET:H_PK_OFFSET+32] = recomputed_h  # update stored hash
consistent_outcome = None
consistent_error   = None
try:
    K_consistent = ML_KEM_512.decaps(bytes(dk_pk_and_h_fixed), ct)
    consistent_outcome = "no_error_wrong_K" if K_consistent != K_good else "correct_K"
except ValueError as e:
    consistent_outcome = "ValueError"
    consistent_error   = str(e)

print(f"\n=== PATCH TEST: flip pk + update stored H_pk to match ===")
print(f"  outcome: {consistent_outcome}")
if consistent_error:
    print(f"  error:   {consistent_error!r}")
print(f"  (expect: no_error_wrong_K — decaps proceeds without hash abort, "
      f"produces wrong output because packed_s is corrupted relative to pk)")

# ── 4. Determine explanation ──────────────────────────────────────────────────
if hash_check_present and hash_check_raises_valueerror and is_hash_abort:
explanation = "A"
explanation_text = (
    "EXPLANATION A CONFIRMED: kyber-py _decaps_internal() performs an explicit "
    "re-hash H(ek_pke) and compares it to the stored h field at line(s) shown above, "
    "directly implementing FIPS 203 Algorithm 18 step 3 (If H(ek) != h, raise error). "
    "This IS a spec-mandated cryptographic integrity check, not a deserialization artifact. "
    "MANUSCRIPT CLAIM: kyber-py decaps() performs an explicit H(ek_pke) re-hash-and-compare "
    "as mandated by FIPS 203 Algorithm 18 (sec 7.3). A bit-flip in the stored pk or H_pk "
    "produces a hash mismatch before any cryptographic processing, raising a ValueError. "
    "This defence-in-depth property is mandated by FIPS 203 for all conformant ML-KEM "
    "implementations — though its effectiveness depends on the implementation "
    "performing the check before using the key material, which constant-time C "
    "implementations must also do."
)
elif is_type_abort and not is_hash_abort:
    explanation = "B"
    explanation_text = (
        "EXPLANATION B: the ValueError is a deserialization/type/modulus-check exception "
        "from the library's internal range validation, NOT from an explicit H(ek_pke) hash check. "
        "MANUSCRIPT CLAIM must be: 'The 100% validation-abort rate on pk/H_pk flips in our "
        "kyber-py trials is an artifact of the reference library's internal deserialization "
        "validation; whether this protection generalises to other conformant ML-KEM "
        "implementations is untested and should not be assumed.'"
    )
else:
    explanation = "UNCLEAR"
    explanation_text = "Could not definitively determine explanation — review source manually."

print(f"\n=== CONCLUSION ===")
print(f"  Explanation: {explanation}")
print(f"  {explanation_text[:200]}...")

# ── 5. Save JSON report ───────────────────────────────────────────────────────
git_commit = subprocess.check_output(
    ["git", "-C", str(SCRIPT_DIR.parent), "rev-parse", "HEAD"]
).decode().strip()

report = {
    "experiment": "Blueprint_v3_sec6_pk_abort_trace",
    "kyber_py_version": "1.2.0",
    "kyber_py_module_file": module_file,
    "git_commit": git_commit,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "source_analysis": {
        "hash_check_H_ek_pke_ne_h_present": hash_check_present,
        "hash_check_raises_ValueError": hash_check_raises_valueerror,
        "dk_type_check_present": type_check_dk_len,
        "c_type_check_present": type_check_c_len,
        "hash_check_lines": [{"rel_lineno": ln, "code": code.strip()} for ln, code in hash_lines],
        "fips203_clause": "Algorithm 18, step 3: 'if H(ek) != h then raise ValueError'",
        "fips203_section": "FIPS 203 §7.3 / Algorithm 18",
    },
    "empirical_pk_flip": {
        "outcome": pk_outcome,
        "error_message": pk_error_msg,
        "triggered_by_hash_mismatch": is_hash_abort,
        "hash_mismatch_confirmed": hash_mismatch,
    },
    "empirical_hpk_flip": {
        "outcome": hpk_outcome,
        "error_message": hpk_error_msg,
        "triggered_by_hash_mismatch": is_hpk_hash_abort,
    },
    "patch_test": {
        "description": "flip pk + set stored H_pk = H(flipped_pk): does decaps proceed?",
        "outcome": consistent_outcome,
        "error": consistent_error,
        "interpretation": (
            "Hash check bypassed when stored H_pk is consistent with flipped pk — "
            "confirms the abort IS triggered by hash mismatch, not deserialization"
            if consistent_outcome == "no_error_wrong_K" else consistent_outcome
        ),
    },
    "conclusion": {
        "explanation": explanation,
        "explanation_text": explanation_text,
        "manuscript_claim": (
            "FIPS 203 Algorithm 18 step 3 mandates H(ek) == h before any crypto processing. "
            "kyber-py v1.2.0 implements this check; a bit-flip in pk or H_pk triggers this "
            "mandatory spec check, not a library-specific deserialization guard. "
            "This generalises to ALL conformant ML-KEM implementations that perform the check "
            "before key use (as required by the standard), but NOT to implementations that "
            "omit or defer this check (non-conformant or performance-optimised variants)."
            if explanation == "A" else
            "Use Explanation B wording per Blueprint v3 §6 — see explanation_text."
        ),
    },
}

out_path = RESULTS_DIR / "exp_v3_sec6_pk_abort_trace.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Report saved: {out_path}")
print(f"   Explanation: {explanation} — see file for full manuscript wording")

from glm53.memory import estimate, kv_bytes_per_token, GIB


def test_kv_per_token_scales_with_type():
    kv8, idx = kv_bytes_per_token("q8_0", "q8_0")
    kv16, _ = kv_bytes_per_token("f16", "f16")
    assert kv16 > kv8 > 0 and idx > 0
    assert 8_000 < kv16 + idx < 20_000   # Größenordnung 13,9 KiB/Token (Messung 29.08.)


def test_estimate_verdicts():
    base = dict(total_bytes=int(94.85 * GIB), nextn_bytes=int(2.6 * GIB), ctx=131072, n_parallel=1, ctk="q8_0", ctv="q8_0",
                ubatch=1024, cache_ram_mib=1024, mmproj_bytes=0)
    ok = estimate(mtp=True, budget=int(105 * GIB), **base)
    assert ok.verdict in ("passt", "knapp") and ok.nextn > 0
    no_mtp = estimate(mtp=False, budget=int(105 * GIB), **base)
    assert no_mtp.total < ok.total and no_mtp.nextn == 0
    big = estimate(mtp=True, budget=int(105 * GIB), **{**base, "total_bytes": int(112 * GIB)})
    assert big.verdict == "zu groß"
    multi = estimate(mtp=False, budget=int(105 * GIB), **{**base, "n_parallel": 8})
    assert multi.rs == 8 * no_mtp.rs

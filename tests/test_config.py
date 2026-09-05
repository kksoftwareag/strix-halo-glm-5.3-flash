from glm53.config import PRESETS, ServerConfig, build_command, get_preset
from glm53.discovery import Inventory, ModelInfo, Engine, Mmproj
from glm53.hardware import Hardware
from glm53.memory import GIB


def inv():
    m = lambda label, arch, src="variant", nextn=int(2.6 * GIB), size=94.85: ModelInfo(
        label=label, repo="unsloth/x", first_shard=f"/m/{label}.{arch}/{label}-00001-of-00004.gguf", n_shards=4,
        total_bytes=int(size * GIB), nextn_bytes=nextn, experts_bytes=int(86 * GIB), arch=arch, source=src)
    return Inventory(
        models=[m("UD-IQ2_XXS", "glm5next", "hf"), m("UD-IQ2_XXS", "glm5-next"), m("UD-Q2_K_XL", "glm5-next", size=101.25),
                m("AJ-IQ3_XXS", "glm5-next", nextn=0, size=104.7)],
        engines=[Engine("tk-mtp", "hip", "/e/tk-mtp/bin"), Engine("tk", "hip", "/e/tk/bin"), Engine("unsloth", "hip", "/e/u/bin")],
        mmproj=[Mmproj("AesSedai F16", "/mm/f16.gguf", "glm5-next", int(1.05 * GIB)), Mmproj("unsloth F16", "/mm/u.gguf", "glm5next", int(1.05 * GIB))],
    )


def hw(avail=105.0):
    return Hardware(mem_total=int(109.7 * GIB), mem_available=int(avail * GIB))


def test_all_presets_build_or_explain():
    for p in PRESETS:
        cmd = build_command(p.apply(), inv(), hw())
        assert cmd.argv[0].endswith("llama-server")
        for e in cmd.errors:
            assert any(s in e for s in ("nicht gefunden", "passt nicht", "Schema", "nicht gebaut")), e


def test_mtp_flags_and_naming():
    cmd = build_command(get_preset("tkmtp-agent").apply(), inv(), hw())
    assert cmd.ok, cmd.errors
    a = cmd.argv
    assert "--spec-type" in a and a[a.index("--spec-type") + 1] == "draft-mtp"
    assert a[a.index("-m") + 1].endswith("UD-IQ2_XXS.glm5-next/UD-IQ2_XXS-00001-of-00004.gguf")
    assert a[a.index("--spec-draft-n-max") + 1] == "2"
    assert "--jinja" in a and '"reasoning_effort": "high"' in a[a.index("--chat-template-kwargs") + 1]
    u = build_command(get_preset("unsloth-agent").apply(), inv(), hw())
    assert u.ok and u.argv[u.argv.index("-m") + 1].endswith("UD-IQ2_XXS.glm5next/UD-IQ2_XXS-00001-of-00004.gguf")


def test_no_mtp_without_nextn_and_plain_engine():
    cfg = ServerConfig(engine="tk-mtp", quant="AJ-IQ3_XXS", mtp_enabled=True)
    assert any("NextN" in e for e in build_command(cfg, inv(), hw()).errors)
    cfg = ServerConfig(engine="tk", quant="UD-IQ2_XXS", mtp_enabled=True)
    assert any("MTP" in e for e in build_command(cfg, inv(), hw()).errors)
    cfg = ServerConfig(engine="tk", quant="UD-IQ2_XXS", mtp_enabled=False)
    cmd = build_command(cfg, inv(), hw())
    assert cmd.ok and "--spec-type" not in cmd.argv


def test_multislot_forces_kv_unified_and_memory_limits():
    cfg = ServerConfig(engine="tk", quant="UD-IQ2_XXS", mtp_enabled=False, n_parallel=4, ctx_size=262144)
    cmd = build_command(cfg, inv(), hw())
    assert "--kv-unified" in cmd.argv and any("kv-unified" in w for w in cmd.warnings)
    cfg = ServerConfig(engine="tk-mtp", quant="UD-Q2_K_XL")
    assert not build_command(cfg, inv(), hw(105)).ok          # 16-GiB-Carve-out: zu groß
    assert build_command(cfg, inv(), hw(120)).ok              # 1-GiB-Carve-out: passt


def test_vision_and_reasoning_validation():
    cmd = build_command(get_preset("chat-vision").apply(), inv(), hw())
    assert cmd.ok and "--mmproj" in cmd.argv and cmd.resolved.mmproj.naming == "glm5-next"
    bad = build_command(ServerConfig(reasoning_effort="medium"), inv(), hw())
    assert any("reasoning_effort" in e for e in bad.errors)

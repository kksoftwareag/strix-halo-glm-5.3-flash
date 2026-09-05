import struct
from pathlib import Path

from glm53.gguf import read_gguf, rewrite_arch, shard_paths, TensorInfo


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path: Path, arch: str, n_tensors: int = 1) -> None:
    kv = [
        (b"general.architecture", 8, _s(arch)),
        (arch.encode() + b".block_count", 4, struct.pack("<I", 46)),
        (arch.encode() + b".attention.head_count_kv", 9, struct.pack("<IQ", 4, 3) + struct.pack("<III", 0, 1, 0)),
        (b"tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _s("a") + _s("bb")),
        (b"general.alignment", 4, struct.pack("<I", 32)),
    ]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, n_tensors, len(kv)))
    for k, t, v in kv:
        out += _s(k.decode()) + struct.pack("<I", t) + v
    for i in range(n_tensors):
        out += _s(f"blk.{i}.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 64, 2) + struct.pack("<IQ", 0, i * 512)
    pad = (-len(out)) % 32
    out += b"\0" * pad
    out += bytes(range(256)) * 2 * n_tensors   # Daten (F32 64x2 = 512 Byte je Tensor)
    path.write_bytes(bytes(out))


def test_read_and_rewrite(tmp_path: Path):
    src = tmp_path / "m-00001-of-00002.gguf"
    make_gguf(src, "glm5next")
    g = read_gguf(src)
    assert g.arch == "glm5next"
    assert g.metadata["glm5next.block_count"] == 46
    assert g.metadata["glm5next.attention.head_count_kv"] == [0, 1, 0]
    assert g.metadata["tokenizer.ggml.tokens#len"] == 2
    assert len(g.tensors) == 1 and g.tensors[0].n_bytes == 512
    dst = tmp_path / "out.gguf"
    info = rewrite_arch(src, dst, "glm5-next")
    assert info["renamed_keys"] == 2
    h = read_gguf(dst)
    assert h.arch == "glm5-next"
    assert h.metadata["glm5-next.block_count"] == 46
    assert h.metadata["glm5-next.attention.head_count_kv"] == [0, 1, 0]
    assert "glm5next.block_count" not in h.metadata
    assert h.metadata["glm5-next.attention.indexer.index_share_mtp"] is True and info["added_keys"] == ["glm5-next.attention.indexer.index_share_mtp"]
    assert h.tensors[0].name == "blk.0.attn_q.weight"
    # Daten unverändert und ausgerichtet
    raw = dst.read_bytes()
    assert raw[-512:] == src.read_bytes()[-512:]
    assert shard_paths(src) == [src, tmp_path / "m-00002-of-00002.gguf"]


def test_tensor_bytes_blockquant():
    t = TensorInfo("x", [256, 4], 16, 0)  # IQ2_XXS: 66 Byte je 256er-Block
    assert t.n_bytes == 66 * 4

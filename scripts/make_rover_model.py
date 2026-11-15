"""Turn a detailed rover model (.glb) into the small mesh the 3D map draws.

    .venv\\Scripts\\python scripts\\make_rover_model.py path\\to\\Perseverance.glb

The source model (NASA's Perseverance, about 200 000 triangles, 24 textures, 11 MB) is far
more than a marker a few pixels across needs, and the page has no glTF loader. This script
reads the .glb itself and writes `mars_map/viewer/rover_mesh.json` (about 200 kB):

* every part is placed by its node transform and merged into one mesh;
* textures are baked into one colour per vertex (sampled at the vertex), with a soft
  top light baked in as well, so the page needs no textures, no lights and no loader;
* the mesh is simplified by vertex clustering: the model is cut into a grid of small
  cubes, all vertices in a cube become one, triangles that collapse are dropped;
* positions are stored as 16-bit integers, colours as bytes, indices as 16-bit.

The result is committed, so nobody needs the 11 MB source to build the pages. Run this
again only to change the model or the level of detail (`--triangles`).

Reads plain glTF 2.0 binaries: float positions, no Draco or meshopt compression.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "mars_map" / "viewer" / "rover_mesh.json"

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
LIGHT = np.array([0.25, 1.0, 0.35]) / np.linalg.norm([0.25, 1.0, 0.35])   # from above, slightly ahead
AMBIENT = 0.58


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, _ = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF" or version != 2:
        raise SystemExit(f"{path} is not a glTF 2.0 binary")
    doc, blob, at = None, b"", 12
    while at < len(raw):
        size, kind = struct.unpack("<I4s", raw[at:at + 8])
        body = raw[at + 8:at + 8 + size]
        if kind == b"JSON":
            doc = json.loads(body)
        elif kind.startswith(b"BIN"):
            blob = body
        at += 8 + size
    if doc is None:
        raise SystemExit("no JSON chunk in the file")
    if doc.get("extensionsRequired"):
        raise SystemExit(f"the model needs extensions this script does not read: {doc['extensionsRequired']}")
    return doc, blob


def accessor(doc: dict, blob: bytes, index: int) -> np.ndarray:
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    dtype, width = np.dtype(COMPONENT[a["componentType"]]), WIDTH[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    if stride == dtype.itemsize * width:
        out = np.frombuffer(blob, dtype, a["count"] * width, start).reshape(a["count"], width)
    else:                                          # interleaved attributes
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(blob, np.uint8, offset=start), (a["count"], dtype.itemsize * width), (stride, 1))
        out = np.ascontiguousarray(rows).view(dtype).reshape(a["count"], width)
    return out.astype(np.float64) if dtype.kind == "f" else out.astype(np.int64)


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] *= np.array(node.get("scale", [1, 1, 1]), float)
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


class Textures:
    """Base-colour textures, shrunk: one colour per vertex needs no fine detail."""

    def __init__(self, doc: dict, blob: bytes):
        self.doc, self.blob, self.cache = doc, blob, {}

    def colours(self, texture_index: int, uv: np.ndarray) -> np.ndarray | None:
        from PIL import Image
        if texture_index not in self.cache:
            image = self.doc["images"][self.doc["textures"][texture_index]["source"]]
            if "bufferView" not in image:
                self.cache[texture_index] = None
            else:
                view = self.doc["bufferViews"][image["bufferView"]]
                data = self.blob[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
                im = Image.open(io.BytesIO(data)).convert("RGB").resize((128, 128), Image.BILINEAR)
                self.cache[texture_index] = np.asarray(im, float) / 255.0
        tex = self.cache[texture_index]
        if tex is None:
            return None
        u = np.floor((uv[:, 0] % 1.0) * 128).astype(int) % 128
        v = np.floor((uv[:, 1] % 1.0) * 128).astype(int) % 128
        return tex[v, u]


def gather(doc: dict, blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All triangles of the scene in one mesh: positions, normals, colours (sRGB 0..1), indices."""
    textures = Textures(doc, blob)
    P, N, C, I, base = [], [], [], [], 0

    def visit(index: int, parent: np.ndarray) -> None:
        nonlocal base
        node = doc["nodes"][index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            rot = world[:3, :3]
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                mat = doc["materials"][prim["material"]] if "material" in prim else {}
                if mat.get("alphaMode") == "BLEND" or "transparent" in mat.get("name", "").lower():
                    continue                       # glass and see-through helper surfaces
                pos = accessor(doc, blob, prim["attributes"]["POSITION"])
                pos = pos @ rot.T + world[:3, 3]
                if "NORMAL" in prim["attributes"]:
                    nrm = accessor(doc, blob, prim["attributes"]["NORMAL"]) @ rot.T
                else:
                    nrm = np.tile([0.0, 1.0, 0.0], (len(pos), 1))
                pbr = mat.get("pbrMetallicRoughness", {})
                factor = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], float) ** (1 / 2.2)   # linear -> sRGB
                col = np.tile(factor, (len(pos), 1))
                tex = pbr.get("baseColorTexture")
                if tex is not None:
                    key = f"TEXCOORD_{tex.get('texCoord', 0)}"
                    if key in prim["attributes"]:
                        sampled = textures.colours(tex["index"], accessor(doc, blob, prim["attributes"][key]))
                        if sampled is not None:
                            col = col * sampled
                idx = (accessor(doc, blob, prim["indices"]).ravel() if "indices" in prim
                       else np.arange(len(pos)))
                P.append(pos); N.append(nrm); C.append(col); I.append(idx[: len(idx) // 3 * 3] + base)
                base += len(pos)
        for child in node.get("children", []):
            visit(child, world)

    for root in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return np.vstack(P), np.vstack(N), np.vstack(C), np.concatenate(I).reshape(-1, 3)


def cluster(P, N, C, T, cells: int):
    """Vertex clustering: one vertex per occupied cube of a `cells`-wide grid."""
    lo, span = P.min(0), (P.max(0) - P.min(0)).max()
    key = np.minimum(np.floor((P - lo) / span * cells).astype(np.int64), cells - 1)
    flat = (key[:, 0] * cells + key[:, 1]) * cells + key[:, 2]
    uniq, inverse = np.unique(flat, return_inverse=True)
    count = np.bincount(inverse, minlength=len(uniq))[:, None]
    mean = lambda a: np.stack([np.bincount(inverse, a[:, k], len(uniq)) for k in range(3)], 1) / count  # noqa: E731
    p, c, n = mean(P), mean(C), mean(N)
    t = inverse[T]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]      # collapsed triangles
    _, first = np.unique(np.sort(t, 1), axis=0, return_index=True)                  # the same triangle twice
    t = t[np.sort(first)]
    used = np.unique(t)
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    return p[used], n[used], c[used], remap[t]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", help="the source model (.glb)")
    ap.add_argument("--triangles", type=int, default=14000, help="most triangles to keep (default 14000)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    doc, blob = read_glb(Path(args.glb))
    P, N, C, T = gather(doc, blob)
    print(f"source: {len(T):,} triangles, {len(P):,} vertices, {len(doc.get('images', []))} images")

    best = None
    for cells in range(24, 260, 4):                # finer grid until the budget is used
        p, n, c, t = cluster(P, N, C, T, cells)
        if len(t) > args.triangles or len(p) > 65535:
            break
        best = (cells, p, n, c, t)
    if best is None:
        raise SystemExit("the triangle budget is too small for this model")
    cells, p, n, c, t = best

    # Bake a soft light from above into the colours: the page draws them as they are.
    length = np.linalg.norm(n, axis=1, keepdims=True)
    unit = np.where(length > 1e-6, n / np.maximum(length, 1e-6), [0.0, 1.0, 0.0])
    shade = AMBIENT + (1 - AMBIENT) * np.clip(unit @ LIGHT, 0, 1) * np.clip(length[:, 0], 0, 1)
    rgb = np.clip(c * shade[:, None] * 1.12, 0, 1)

    # Centre it on the ground under its middle; store as 16-bit steps of the largest extent.
    lo, hi = p.min(0), p.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    size = hi - lo
    q = np.round((p - centre) / size.max() * 32767).astype(np.int16)

    def b64(a: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")

    out = {
        "source": Path(args.glb).name,
        "note": "simplified by scripts/make_rover_model.py: vertex clustering, colours and light baked per vertex",
        "vertices": int(len(p)), "triangles": int(len(t)), "grid": cells,
        "size_m": [round(float(v), 3) for v in size],          # x, y (up), z of the source, metres
        "unit": float(size.max() / 32767),                     # metres per integer step
        "pos": b64(q), "col": b64(np.round(rgb * 255).astype(np.uint8)), "idx": b64(t.astype(np.uint16)),
    }
    path = Path(args.out)
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"kept:   {len(t):,} triangles, {len(p):,} vertices (grid {cells})")
    print(f"size:   {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m (x, up, z)")
    print(f"wrote:  {path}  ({path.stat().st_size / 1024:.0f} kB, source {Path(args.glb).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

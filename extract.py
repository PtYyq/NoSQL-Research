import argparse
import gzip
import io
import os
import random
from typing import Iterable, Tuple, List, Set

def stream_edges(path: str, snap_flip: bool = False) -> Iterable[Tuple[str, str]]:
    """
    Yield edges as (follower, followee).
    - If snap_flip=False: expect CSV "follower,followee"
    - If snap_flip=True: expect SNAP raw "i j" meaning edge (j -> i)
    """
    # auto-detect gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if snap_flip:
                # SNAP raw: skip comments
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                i, j = parts
                follower, followee = j, i  # flip to j -> i
            else:
                # CSV: follower,followee (no header assumed)
                parts = line.split(",")
                if len(parts) != 2:
                    # tolerate a header like "follower,followee"
                    if line.lower().replace(" ", "") in ("follower,followee",):
                        continue
                    else:
                        continue
                follower, followee = parts[0].strip(), parts[1].strip()
            if follower and followee:
                yield (follower, followee)

def reservoir_sample(edges: Iterable[Tuple[str, str]], k: int, seed: int) -> List[Tuple[str, str]]:
    """
    Standard reservoir sampling over an unknown-length stream.
    Keeps exactly k edges uniformly at random (or all if < k).
    """
    random.seed(seed)
    R: List[Tuple[str, str]] = []
    t = 0
    for e in edges:
        t += 1
        if t <= k:
            R.append(e)
        else:
            j = random.randint(1, t)
            if j <= k:
                R[j - 1] = e
    return R

def write_edges(path: str, edges: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fo:
        for u, v in edges:
            fo.write(f"{u},{v}\n")

def write_rels_with_header(path: str, edges: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fo:
        fo.write("follower,followee\n")
        for u, v in edges:
            fo.write(f"{u},{v}\n")

def write_nodes(path: str, edges: List[Tuple[str, str]]) -> None:
    seen: Set[str] = set()
    for u, v in edges:
        seen.add(u); seen.add(v)
    with open(path, "w", encoding="utf-8", newline="") as fo:
        fo.write("uid\n")
        for uid in sorted(seen, key=lambda x: (len(x), x)):
            fo.write(f"{uid}\n")

def main():
    ap = argparse.ArgumentParser(description="Reservoir sampling for Twitter edges.")
    ap.add_argument("-i", "--input", required=True, help="Input file path (CSV or SNAP .gz)")
    ap.add_argument("-k", "--sample-size", type=int, required=True, help="Number of edges to sample")
    ap.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--snap-flip", action="store_true",
                   help="Input is SNAP raw (i j means j follows i); flip to follower=j,followee=i")
    ap.add_argument("-o", "--output-sample", default="edges_sample.csv",
                   help="Output sampled edges CSV (no header)")
    ap.add_argument("--emit-rels", metavar="RELS_CSV",
                   help="Also write rels CSV with header 'follower,followee'")
    ap.add_argument("--emit-nodes", metavar="NODES_CSV",
                   help="Also write nodes CSV with header 'uid' (unique endpoints)")
    args = ap.parse_args()

    edges_iter = stream_edges(args.input, snap_flip=args.snap_flip)
    sample = reservoir_sample(edges_iter, args.sample_size, args.seed)

    # Always write raw sampled edges (no header)
    write_edges(args.output_sample, sample)
    print(f"[OK] Sampled {len(sample)} edges -> {args.output_sample}")

    if args.emit_rels:
        write_rels_with_header(args.emit_rels, sample)
        print(f"[OK] Wrote rels (with header) -> {args.emit_rels}")

    if args.emit_nodes:
        write_nodes(args.emit_nodes, sample)
        print(f"[OK] Wrote nodes (unique endpoints) -> {args.emit_nodes}")

if __name__ == "__main__":
    main()


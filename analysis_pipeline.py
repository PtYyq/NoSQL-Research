from __future__ import annotations
import argparse, csv, os, math, time, random
from statistics import median
from typing import List, Dict, Tuple
from neo4j import GraphDatabase

# -------------------- utils --------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def p_quantiles(samples: List[float], q: float):
    if not samples:
        return None
    xs = sorted(samples)
    idx = int(round(q * (len(xs) - 1)))
    return xs[idx]


def clear_query_caches(session):
    try:
        session.run("CALL db.clearQueryCaches();")
    except Exception:
        pass


# -------------------- dataset profiling --------------------

def dataset_counts(session) -> Tuple[int, int]:
    n = session.run("MATCH (u:User) RETURN count(u) AS n").single()["n"]
    m = session.run("MATCH ()-[:FOLLOWS]->() RETURN count(*) AS m").single()["m"]
    return int(n), int(m)


def degree_histogram(session, direction: str) -> List[Tuple[int,int]]:
    assert direction in ("in", "out")
    if direction == "in":
        q = (
            "MATCH (u:User)\n"
            "OPTIONAL MATCH (u)<-[r:FOLLOWS]-()\n"
            "WITH u, count(r) AS deg\n"
            "RETURN deg AS degree, count(*) AS users\n"
            "ORDER BY degree"
        )
    else:
        q = (
            "MATCH (u:User)\n"
            "OPTIONAL MATCH (u)-[r:FOLLOWS]->()\n"
            "WITH u, count(r) AS deg\n"
            "RETURN deg AS degree, count(*) AS users\n"
            "ORDER BY degree"
        )
    rows = [(int(r["degree"]), int(r["users"])) for r in session.run(q)]
    return rows


def histogram_to_percentiles(hist: List[Tuple[int,int]], ps=(0.5, 0.95)) -> Dict[float, int]:
    total = sum(c for _, c in hist)
    if total == 0:
        return {p: 0 for p in ps}
    cume = 0
    res = {}
    targets = sorted(ps)
    ti = 0
    for deg, cnt in hist:
        cume += cnt
        while ti < len(targets) and cume / total >= targets[ti]:
            res[targets[ti]] = deg
            ti += 1
        if ti >= len(targets):
            break
    return res


def histogram_to_ccdf(hist: List[Tuple[int,int]]) -> List[Tuple[int,float]]:
    total = sum(c for _, c in hist)
    if total == 0:
        return []
    # cumulative from the tail
    hist_sorted = sorted(hist, key=lambda x: x[0])
    suffix = 0
    ccdf = []
    # compute counts >= degree for each degree present
    degrees = [d for d,_ in hist_sorted]
    counts = [c for _,c in hist_sorted]
    # precompute suffix sums
    suffix_counts = [0]*len(counts)
    run = 0
    for i in range(len(counts)-1, -1, -1):
        run += counts[i]
        suffix_counts[i] = run
    for i, deg in enumerate(degrees):
        ccdf.append((deg, suffix_counts[i]/total))
    return ccdf


def write_csv(path: str, header: List[str], rows: List[Tuple]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


# -------------------- utility selections --------------------

def sample_user_ids(session, n=50) -> List[str]:
    # random sample using rand() ordering
    rows = session.run("MATCH (u:User) WITH u, rand() AS r RETURN u.uid AS uid ORDER BY r LIMIT $n", n=n)
    return [r["uid"] for r in rows]


def sample_mutual_pairs(session, n=50) -> List[Tuple[str,str]]:
    q = (
        "MATCH (a:User)-[:FOLLOWS]->(b:User) "
        "MATCH (b)-[:FOLLOWS]->(a) WHERE id(a) < id(b) "
        "RETURN a.uid AS a, b.uid AS b LIMIT $n"
    )
    return [(r["a"], r["b"]) for r in session.run(q, n=n)]


# -------------------- benchmark kernels --------------------

def timed_run(session, query: str, params: Dict, cold: bool) -> float:
    if cold:
        clear_query_caches(session)
    t0 = time.perf_counter()
    list(session.run(query, **params))
    return (time.perf_counter() - t0) * 1000.0


def bench_followers(session, uid: str, page: int, repeats: int, cold: bool) -> List[float]:
    q = (
        "MATCH (u:User {uid:$uid})<-[:FOLLOWS]-(f:User) "
        "RETURN f.uid AS follower ORDER BY follower SKIP 0 LIMIT $lim"
    )
    durs = []
    for _ in range(repeats):
        durs.append(timed_run(session, q, {"uid": uid, "lim": page}, cold))
    return durs


def bench_mutual(session, a: str, b: str, repeats: int, cold: bool) -> List[float]:
    q = (
        "MATCH (a:User {uid:$a})-[:FOLLOWS]->(b:User {uid:$b}) "
        "MATCH (b)-[:FOLLOWS]->(a) RETURN true LIMIT 1"
    )
    durs = []
    for _ in range(repeats):
        durs.append(timed_run(session, q, {"a": a, "b": b}, cold))
    return durs


def bench_foaf(session, uid: str, k: int, repeats: int, cold: bool) -> List[float]:
    q = (
        "MATCH (me:User {uid:$uid})-[:FOLLOWS]->(:User)-[:FOLLOWS]->(cand:User) "
        "WHERE me <> cand AND NOT EXISTS { MATCH (me)-[:FOLLOWS]->(cand) } "
        "RETURN cand.uid AS uid, count(*) AS score ORDER BY score DESC, uid DESC LIMIT $k"
    )
    durs = []
    for _ in range(repeats):
        durs.append(timed_run(session, q, {"uid": uid, "k": k}, cold))
    return durs


def bench_time_window(session, uid: str, t0_str: str, t1_str: str, k: int, repeats: int, cold: bool) -> List[float]:
    q = (
        "MATCH (me:User {uid:$uid})-[r:FOLLOWS]->(u:User) "
        "WHERE r.createdAt >= datetime($t0) AND r.createdAt < datetime($t1) "
        "RETURN u.uid AS uid ORDER BY r.createdAt DESC LIMIT $k"
    )
    durs = []
    for _ in range(repeats):
        durs.append(timed_run(session, q, {"uid": uid, "t0": t0_str, "t1": t1_str, "k": k}, cold))
    return durs


# -------------------- FOAF candidate counts --------------------

def foaf_candidate_counts(session) -> List[Tuple[str,int]]:
    q = (
        "MATCH (me:User)-[:FOLLOWS]->(:User)-[:FOLLOWS]->(cand:User) "
        "WHERE me <> cand AND NOT EXISTS { MATCH (me)-[:FOLLOWS]->(cand) } "
        "WITH me, count(DISTINCT cand) AS candCount "
        "RETURN me.uid AS uid, candCount ORDER BY candCount DESC, uid"
    )
    rows = [(r["uid"], int(r["candCount"])) for r in session.run(q)]
    return rows


# -------------------- LaTeX helpers --------------------
def write_latex_table_latency(path: str, rows: list[tuple[str,str,float,float,float,str]]):
    with open(path, 'w') as f:
        f.write("""%% Auto-generated latency table
\\begin{table}[t]
\\centering
\\caption{Latency (ms) under %s cache (median of %d runs)}
\\label{tab:latency}
\\begin{tabular}{lrrrr}
\\hline
Query & P50 & P95 & Max & Notes \\\\
\\hline
""" % ("warm/cold", 5))
        for q, variant, p50, p95, mx, notes in rows:
            f.write(f"{q} ({variant}) & {p50:.2f} & {p95:.2f} & {mx:.2f} & {notes} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def write_latex_table_dataset(path: str, n: int, m: int,
                              indeg_p50: int, indeg_p95: int, indeg_max: int,
                              outdeg_p50: int, outdeg_p95: int, outdeg_max: int):
    with open(path, 'w') as f:
        f.write("""%% Auto-generated dataset profile table
\\begin{table}[t]
\\centering
\\caption{Dataset statistics (sampled subgraph)}
\\label{tab:data}
\\begin{tabular}{lrrrr}
\\hline
Metric & Value & P50 & P95 & Max \\\\
\\hline
Nodes $|V|$ & %d & -- & -- & -- \\\\
Edges $|E|$ & %d & -- & -- & -- \\\\
In-degree & -- & %d & %d & %d \\\\
Out-degree & -- & %d & %d & %d \\\\
\\hline
\\end{tabular}
\\end{table}
""" % (n, m, indeg_p50, indeg_p95, indeg_max, outdeg_p50, outdeg_p95, outdeg_max))


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--uids", nargs="*", default=[])
    ap.add_argument("--page", type=int, default=50)
    ap.add_argument("--foaf_k", type=int, default=10)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--t0")
    ap.add_argument("--t1")
    ap.add_argument("--outdir", default="analysis_out")

    args = ap.parse_args()
    ensure_outdir(args.outdir)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as session:
        # Dataset profile
        n, m = dataset_counts(session)
        indeg_hist = degree_histogram(session, "in")
        outdeg_hist = degree_histogram(session, "out")
        indeg_p = histogram_to_percentiles(indeg_hist, ps=(0.5, 0.95))
        outdeg_p = histogram_to_percentiles(outdeg_hist, ps=(0.5, 0.95))
        indeg_ccdf = histogram_to_ccdf(indeg_hist)
        outdeg_ccdf = histogram_to_ccdf(outdeg_hist)

        write_csv(os.path.join(args.outdir, "degree_hist_indeg.csv"), ["degree","users"], indeg_hist)
        write_csv(os.path.join(args.outdir, "degree_hist_outdeg.csv"), ["degree","users"], outdeg_hist)
        write_csv(os.path.join(args.outdir, "ccdf_indeg.csv"), ["degree","ccdf"], indeg_ccdf)
        write_csv(os.path.join(args.outdir, "ccdf_outdeg.csv"), ["degree","ccdf"], outdeg_ccdf)
        write_csv(os.path.join(args.outdir, "dataset_profile.csv"), ["metric","value"], [
            ("nodes", n), ("edges", m),
            ("indeg_p50", indeg_p.get(0.5, 0)), ("indeg_p95", indeg_p.get(0.95, 0)), ("indeg_max", indeg_hist[-1][0] if indeg_hist else 0),
            ("outdeg_p50", outdeg_p.get(0.5, 0)), ("outdeg_p95", outdeg_p.get(0.95, 0)), ("outdeg_max", outdeg_hist[-1][0] if outdeg_hist else 0),
        ])
        write_latex_table_dataset(
            os.path.join(args.outdir, "latex_table_dataset.tex"),
            n, m,
            indeg_p.get(0.5, 0), indeg_p.get(0.95, 0), indeg_hist[-1][0] if indeg_hist else 0,
            outdeg_p.get(0.5, 0), outdeg_p.get(0.95, 0), outdeg_hist[-1][0] if outdeg_hist else 0,
        )

        # FOAF candidate distribution (optional but quick)
        try:
            foaf_rows = foaf_candidate_counts(session)
            write_csv(os.path.join(args.outdir, "foaf_candidates.csv"), ["uid","candCount"], foaf_rows)
        except Exception as e:
            # If graph is tiny, this may be empty; ignore
            pass

        # Benchmark selections
        uids = args.uids or sample_user_ids(session, n=20)
        if not uids:
            print("[warn] No users found to benchmark.")
        pairs = sample_mutual_pairs(session, n=20)

        # Benchmarks aggregated across sampled users
        bench_rows: List[Tuple[str,str,float,float,float,str]] = []

        # Followers pagination (run on first uid as representative)
        if uids:
            uid0 = uids[0]
            d = []
            for _ in range(args.repeats):
                d.extend(bench_followers(session, uid0, page=args.page, repeats=1, cold=args.cold))
            bench_rows.append(("followers", f"page={args.page}", median(d), p_quantiles(d,0.95), max(d), f"uid={uid0}"))

        # Mutual follow (random pairs)
        if pairs:
            d = []
            for _ in range(args.repeats):
                a,b = random.choice(pairs)
                d.extend(bench_mutual(session, a, b, repeats=1, cold=args.cold))
            bench_rows.append(("mutual", "--", median(d), p_quantiles(d,0.95), max(d), f"pairs={len(pairs)}"))

        # FOAF Top-k (run on first uid)
        if uids:
            uid0 = uids[0]
            d = []
            for _ in range(args.repeats):
                d.extend(bench_foaf(session, uid0, k=args.foaf_k, repeats=1, cold=args.cold))
            bench_rows.append(("foaf", f"k={args.foaf_k}", median(d), p_quantiles(d,0.95), max(d), f"uid={uid0}"))

        # Time-window (if provided)
        if uids and args.t0 and args.t1:
            uid0 = uids[0]
            try:
                d = []
                for _ in range(args.repeats):
                    d.extend(bench_time_window(session, uid0, args.t0, args.t1, k=args.page, repeats=1, cold=args.cold))
                bench_rows.append(("time_window", f"k={args.page}", median(d), p_quantiles(d,0.95), max(d), f"uid={uid0}"))
            except Exception as e:
                print(f"[warn] time-window query failed: {e}")

        # Write bench CSV + LaTeX
        write_csv(os.path.join(args.outdir, "bench_results.csv"), ["query","variant","p50_ms","p95_ms","max_ms","notes"], bench_rows)
        write_latex_table_latency(os.path.join(args.outdir, "latex_table_latency.tex"), bench_rows)

    driver.close()
    print(f"[done] outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()

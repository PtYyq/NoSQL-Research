from __future__ import annotations
import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Set

try:
    from neo4j import GraphDatabase
except Exception as e:  # pragma: no cover
    print("[ERROR] Missing dependency 'neo4j'. Install via: pip install neo4j", file=sys.stderr)
    raise


# -----------------------------
# Neo4j helpers
# -----------------------------

def get_all_users_and_edges(driver) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Fetch all User uids and FOLLOWS edges (src_uid, dst_uid).
    Ensures nodes with zero degree are included.
    """
    users: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    with driver.session() as session:
        # Stream edges (avoid collecting huge lists inside Neo4j)
        for rec in session.run(
            """
            MATCH (a:User)-[:FOLLOWS]->(b:User)
            RETURN a.uid AS src, b.uid AS dst
            """
        ):
            src, dst = rec["src"], rec["dst"]
            edges.append((src, dst))
            users.add(src)
            users.add(dst)

        # Include isolated users as well
        for rec in session.run("MATCH (u:User) RETURN u.uid AS uid"):
            users.add(rec["uid"])

    return users, edges


def get_two_hop_candidates(driver, me_uid: str) -> Set[str]:
    """Return set of 2-hop candidates for 'me_uid' that 'me' doesn't already follow."""
    with driver.session() as session:
        res = session.run(
            """
            MATCH (me:User {uid:$uid})-[:FOLLOWS]->(:User)-[:FOLLOWS]->(cand:User)
            WHERE me <> cand AND NOT (me)-[:FOLLOWS]->(cand)
            RETURN DISTINCT cand.uid AS uid
            """,
            uid=me_uid,
        )
        return {r["uid"] for r in res}


# -----------------------------
# PageRank (pure Python)
# -----------------------------

def pagerank(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    personalization: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Compute (Personalized) PageRank via power iteration.

    Args:
        nodes: node identifiers (strings)
        edges: directed edges (src, dst)
        damping: damping factor (alpha)
        max_iter: max iterations
        tol: L1 tolerance for convergence
        personalization: optional dict mapping node->weight (will be normalized)
                         If None, uniform distribution is used

    Returns:
        dict of node->rank (sums to 1)
    """
    nodes = list(nodes)
    if not nodes:
        return {}

    N = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    out_neighbors: List[List[int]] = [[] for _ in range(N)]
    out_deg = [0] * N

    for s, t in edges:
        if s not in idx or t not in idx:
            # Skip edges to/from nodes not in the nodes set
            continue
        si, ti = idx[s], idx[t]
        out_neighbors[si].append(ti)
        out_deg[si] += 1

    # Personalization vector v (teleport), and dangling distribution s
    if personalization:
        v = [0.0] * N
        total = sum(max(0.0, personalization.get(u, 0.0)) for u in nodes)
        if total == 0:
            # Fall back to uniform if ill-specified
            v = [1.0 / N] * N
        else:
            for u, i in idx.items():
                v[i] = max(0.0, personalization.get(u, 0.0)) / total
    else:
        v = [1.0 / N] * N

    # For dangling nodes, we redistribute their rank according to v (standard choice)
    r = [1.0 / N] * N  # initial rank

    for _ in range(max_iter):
        new_r = [0.0] * N
        # Distribute rank along outgoing links
        for i in range(N):
            if out_deg[i] == 0:
                continue
            contrib = r[i] / out_deg[i]
            for j in out_neighbors[i]:
                new_r[j] += damping * contrib

        # Handle dangling mass
        dangling_mass = sum(r[i] for i in range(N) if out_deg[i] == 0)
        if dangling_mass:
            dm = damping * dangling_mass
            for i in range(N):
                new_r[i] += dm * v[i]

        # Teleportation
        one_minus = 1.0 - damping
        for i in range(N):
            new_r[i] += one_minus * v[i]

        # L1 diff
        diff = sum(abs(new_r[i] - r[i]) for i in range(N))
        r = new_r
        if diff <= tol:
            break

    # Normalize (precaution)
    s = sum(r)
    if s > 0:
        r = [x / s for x in r]

    return {u: r[idx[u]] for u in nodes}


# -----------------------------
# Optional: Neo4j GDS PageRank
# -----------------------------

def gds_pagerank(driver, top: int = 20) -> List[Tuple[str, float]]:
    """Try to run Neo4j GDS PageRank; raises if GDS isn't available."""
    with driver.session() as session:
        try:
            result = session.run(
                """
                CALL gds.pageRank.stream({
                  nodeProjection: 'User',
                  relationshipProjection: { FOLLOWS: { type: 'FOLLOWS', orientation: 'NATURAL' } }
                })
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).uid AS uid, score
                ORDER BY score DESC
                LIMIT $top
                """,
                top=top,
            )
        except Exception as e:
            raise RuntimeError("GDS PageRank not available or query failed") from e

        rows = [(rec["uid"], float(rec["score"])) for rec in result]
        return rows


# -----------------------------
# CLI workflows
# -----------------------------

def do_global(driver, use_gds: bool, top: int):
    t0 = time.time()
    if use_gds:
        try:
            rows = gds_pagerank(driver, top=top)
            print("[info] Using GDS PageRank")
            for uid, score in rows:
                print(f"{uid}\t{score:.8f}")
            print(f"[done] {len(rows)} rows in {time.time()-t0:.3f}s")
            return
        except Exception as e:
            print(f"[warn] GDS failed: {e}; falling back to pure Python…")

    users, edges = get_all_users_and_edges(driver)
    ranks = pagerank(users, edges)
    rows = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top]
    for uid, score in rows:
        print(f"{uid}\t{score:.8f}")
    print(f"[done] {len(rows)} rows in {time.time()-t0:.3f}s")


def do_personalized(driver, seeds: List[str], top: int):
    if not seeds:
        print("[ERROR] --seed-uids required for personalized mode", file=sys.stderr)
        sys.exit(2)
    users, edges = get_all_users_and_edges(driver)
    pers = {u: (1.0 if u in seeds else 0.0) for u in users}
    t0 = time.time()
    ranks = pagerank(users, edges, personalization=pers)
    rows = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top]
    for uid, score in rows:
        print(f"{uid}\t{score:.8f}")
    print(f"[done] {len(rows)} rows in {time.time()-t0:.3f}s")


def do_recommend(driver, me_uid: str, top: int):
    if not me_uid:
        print("[ERROR] --seed-uids exactly one uid required for recommend mode", file=sys.stderr)
        sys.exit(2)

    users, edges = get_all_users_and_edges(driver)
    if me_uid not in users:
        print(f"[ERROR] user '{me_uid}' not found in graph", file=sys.stderr)
        sys.exit(2)

    # Personalized PR centered at me_uid
    pers = {u: 0.0 for u in users}
    pers[me_uid] = 1.0
    ranks = pagerank(users, edges, personalization=pers)

    # Candidate set = 2-hop non-followed
    cand_set = get_two_hop_candidates(driver, me_uid)
    if not cand_set:
        print("[info] No candidates found (2-hop).")
        return

    # Rank candidates by PPR score
    rows = [(uid, ranks.get(uid, 0.0)) for uid in cand_set]
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top]

    print(f"Recommendations for '{me_uid}' (Personalized PageRank):")
    for uid, score in rows:
        print(f"{uid}\t{score:.8f}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Neo4j Page Analysis (PageRank)")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--mode", choices=["global", "personalized", "recommend"], default="global")
    parser.add_argument("--seed-uids", nargs="*", default=[], help="User ids for personalization or recommend (use one uid for recommend)")
    parser.add_argument("--use-gds", action="store_true", help="Try Neo4j GDS PageRank first, fallback to Python if unavailable")

    args = parser.parse_args()

    if args.mode == "recommend":
        if len(args.seed_uids) != 1:
            print("[ERROR] recommend mode requires exactly one --seed-uids value (the user to recommend for)", file=sys.stderr)
            sys.exit(2)

    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
        # Test connection
        with driver.session() as s:
            s.run("RETURN 1")
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Failed to connect to Neo4j at {args.uri}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.mode == "global":
            do_global(driver, use_gds=args.use_gds, top=args.top)
        elif args.mode == "personalized":
            do_personalized(driver, seeds=args.seed_uids, top=args.top)
        elif args.mode == "recommend":
            do_recommend(driver, me_uid=args.seed_uids[0], top=args.top)
    finally:
        driver.close()


if __name__ == "__main__":
    main()

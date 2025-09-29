# NoSQL-Research

## extract.py
Input formats supported:
- CSV (no header): each line "follower,followee"
- SNAP raw .gz: lines "i j" meaning j follows i  (use --snap-flip to output follower=j,followee=i)

Outputs:
- A sampled edges file (default: edges_sample.csv)
- Optional rels.csv with header "follower,followee"
- Optional nodes.csv with header "uid" (unique endpoints from the sampled edges)

Usage examples:
```
python sample_edges.py -i edges_follow.csv -k 200000 \
      -o edges_sample.csv --emit-rels rels.csv --emit-nodes nodes.csv
```

## page_rank.py
This script connects to a Neo4j database and computes:
  1) Global PageRank over (:User)-[:FOLLOWS]->(:User) graph
  2) Personalized PageRank for given seed user(s)
  3) (Optional) A FOAF-style recommendation list ranked by Personalized PageRank

Usage examples:
  ### Global PageRank, top 20
  ```python neo4j_pagerank.py --top 20```

  ### Personalized PageRank for one user
  ```python neo4j_pagerank.py --mode personalized --seed-uids alice --top 20```

  ### Recommend candidates for a user (2-hop non-followed), ranked by PPR
  ```python neo4j_pagerank.py --mode recommend --seed-uids alice --top 10```

  ### Use explicit connection flags
  ```python neo4j_pagerank.py --uri bolt://localhost:7687 --user neo4j --password secret```

## analysis_pipeline.py
Generates experimental data.

Usage examples:
```python neo4j_analysis_pipeline.py --user neo4j --password '***' --repeats 7```

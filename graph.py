"""Biomedical knowledge graph used for graph aware retrieval and explanations."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from medfusion.utils import neg


class KnowledgeGraph:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        self.graph = nx.DiGraph()
        for row in nodes.itertuples(index=False):
            self.graph.add_node(row.node_id, label=row.label, node_type=row.node_type)
        for row in edges.itertuples(index=False):
            self.graph.add_edge(row.source, row.target, relation=row.relation, weight=float(row.weight),
                                evidence=row.evidence_doc_id)
        self.undirected = self.graph.to_undirected(as_view=True)

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        return cls(pd.read_csv(directory / "kg_nodes.csv"), pd.read_csv(directory / "kg_edges.csv"))

    def label(self, node: str):
        return self.graph.nodes[node]["label"] if node in self.graph else node

    def neighbours(self, node: str):
        if node not in self.graph:
            return []
        out = [(t, d["weight"]) for _, t, d in self.graph.out_edges(node, data=True)]
        out += [(s, d["weight"]) for s, _, d in self.graph.in_edges(node, data=True)]
        return out

    def evidence_paths(self, sources: list[str], target: str, max_hops: int = 3):
        """Strongest directed reasoning chains from observed concepts to a conclusion.

        Edge cost is the negative log of its weight, so the shortest path is the
        most plausible chain of evidence.
        """
        import math
        paths = []
        if target not in self.graph:
            return paths
        for source in sources:
            if source not in self.graph or source == target:
                continue
            try:
                path = nx.shortest_path(self.graph, source, target,
                                        weight=lambda u, v, d: neg(math.log(max(d["weight"], 0.01))))
            except nx.NetworkXNoPath:
                continue
            if len(path) > max_hops + 1:
                continue
            steps, strength = [], 1.0
            for u, v in zip(path[:~0], path[1:]):
                data = self.graph.edges[u, v]
                strength *= data["weight"]
                steps.append({"from": self.label(u), "relation": data["relation"].replace("_", " "),
                              "to": self.label(v), "evidence": data["evidence"]})
            paths.append({"source": self.label(source), "target": self.label(target),
                          "strength": round(strength, 3), "steps": steps})
        return sorted(paths, key=lambda p: p["strength"], reverse=True)

    def summary(self):
        types = pd.Series([d["node_type"] for _, d in self.graph.nodes(data=True)]).value_counts().to_dict()
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges(), "node_types": types}

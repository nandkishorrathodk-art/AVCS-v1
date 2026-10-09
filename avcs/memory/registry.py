"""Persistent UI Registry with Hungarian Re-ID.

Maintains entity identities and lifecycles across viewport scrolling, window resizing,
and dynamic UI re-renders, preventing coordinate drift across multi-turn agent sessions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from avcs.memory.types import EntityLifecycle, UIMLElement


class PersistentUIRegistry:
    """Tracks UI elements persistently across multiple observation turns using bipartite matching."""

    def __init__(self, cost_threshold: float = 0.40) -> None:
        self.cost_threshold = cost_threshold
        self.registry: Dict[str, UIMLElement] = {}
        self.lifecycles: Dict[str, EntityLifecycle] = {}
        self.interaction_counts: Dict[str, int] = {}
        self._next_id_counter = 1

    def update(
        self,
        observed_elements: List[UIMLElement],
        container_flow: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[List[UIMLElement], List[str]]:
        """Matches observed elements against registry, preserving IDs and lifecycles.

        Returns:
            (active_elements, disappeared_ids)
        """
        existing_ids = [
            eid for eid, life in self.lifecycles.items()
            if life in (EntityLifecycle.VISIBLE, EntityLifecycle.OCCLUDED)
        ]

        if not existing_ids:
            # First turn: register all observed elements
            active: List[UIMLElement] = []
            for elem in observed_elements:
                pid = f"#e{self._next_id_counter}"
                self._next_id_counter += 1
                elem.element_id = pid
                self.registry[pid] = elem
                self.lifecycles[pid] = EntityLifecycle.VISIBLE
                self.interaction_counts[pid] = 0
                active.append(elem)
            return active, []

        N_old = len(existing_ids)
        N_new = len(observed_elements)
        dx, dy = container_flow

        # 1. Build Affinity Cost Matrix C (N_old x N_new)
        C = np.zeros((N_old, N_new), dtype=np.float32)
        for i, old_id in enumerate(existing_ids):
            old_elem = self.registry[old_id]
            old_cx, old_cy = old_elem.bbox.center
            # Flow-compensated expected centroid
            exp_cx = old_cx + dx
            exp_cy = old_cy + dy

            for j, new_elem in enumerate(observed_elements):
                new_cx, new_cy = new_elem.bbox.center

                # Spatial euclidean distance
                spatial_dist = math.hypot(exp_cx - new_cx, exp_cy - new_cy)

                # Role consistency cost
                role_cost = 0.0 if old_elem.role == new_elem.role else 0.35

                # Label string distance
                label_cost = 0.0 if old_elem.label.lower() == new_elem.label.lower() else 0.25

                # Visual embedding cosine distance if present
                vis_cost = 0.0
                if old_elem.visual_embedding is not None and new_elem.visual_embedding is not None:
                    denom = (
                        np.linalg.norm(old_elem.visual_embedding)
                        * np.linalg.norm(new_elem.visual_embedding)
                        + 1e-8
                    )
                    cos_sim = float(np.dot(old_elem.visual_embedding, new_elem.visual_embedding) / denom)
                    vis_cost = (1.0 - cos_sim) * 0.3

                total_cost = 0.50 * spatial_dist + role_cost + label_cost + vis_cost
                C[i, j] = total_cost

        # 2. Solve Optimal Assignment (Hungarian / greedy minimum cost matching)
        row_ind, col_ind = self._solve_assignment(C)

        matched_old: Set[int] = set()
        matched_new: Set[int] = set()
        active_elements: List[UIMLElement] = []

        for r, c in zip(row_ind, col_ind):
            if C[r, c] < self.cost_threshold:
                old_id = existing_ids[r]
                elem = observed_elements[c]
                # Re-assign persistent ID
                elem.element_id = old_id
                self.registry[old_id] = elem
                self.lifecycles[old_id] = EntityLifecycle.VISIBLE
                matched_old.add(r)
                matched_new.add(c)
                active_elements.append(elem)

        # 3. Handle Unmatched Old Entities (Scrolled off-screen or destroyed)
        disappeared_ids: List[str] = []
        for r in range(N_old):
            if r not in matched_old:
                old_id = existing_ids[r]
                # Check if movement indicates scroll
                if dy > 0.05:
                    self.lifecycles[old_id] = EntityLifecycle.OFF_SCREEN_BELOW
                elif dy < -0.05:
                    self.lifecycles[old_id] = EntityLifecycle.OFF_SCREEN_ABOVE
                else:
                    self.lifecycles[old_id] = EntityLifecycle.DESTROYED
                    disappeared_ids.append(old_id)

        # 4. Handle Unmatched New Entities (Spawned elements)
        for c in range(N_new):
            if c not in matched_new:
                elem = observed_elements[c]
                new_id = f"#e{self._next_id_counter}"
                self._next_id_counter += 1
                elem.element_id = new_id
                self.registry[new_id] = elem
                self.lifecycles[new_id] = EntityLifecycle.VISIBLE
                self.interaction_counts[new_id] = 0
                active_elements.append(elem)

        # Clean up destroyed items from active table
        for did in disappeared_ids:
            if did in self.registry:
                del self.registry[did]

        return active_elements, disappeared_ids

    def _solve_assignment(self, C: np.ndarray) -> Tuple[List[int], List[int]]:
        """Solves bipartite assignment via Hungarian algorithm (or greedy fallback)."""
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            r, c = linear_sum_assignment(C)
            return list(r), list(c)
        except Exception:
            # Zero-dependency greedy assignment fallback
            rows, cols = C.shape
            matched_r, matched_c = [], []
            used_cols = set()
            # Sort edges by increasing cost
            edges = []
            for r in range(rows):
                for col in range(cols):
                    edges.append((C[r, col], r, col))
            edges.sort(key=lambda x: x[0])

            used_rows = set()
            for cost, r, col in edges:
                if r not in used_rows and col not in used_cols:
                    if cost < self.cost_threshold:
                        matched_r.append(r)
                        matched_c.append(col)
                        used_rows.add(r)
                        used_cols.add(col)
            return matched_r, matched_c

    def get_element(self, element_id: str) -> Optional[UIMLElement]:
        return self.registry.get(element_id)

    def record_interaction(self, element_id: str) -> None:
        if element_id in self.interaction_counts:
            self.interaction_counts[element_id] += 1

"""
workflow_optimizer.py

Profile-Guided Workflow Optimizer
---------------------------------

Implements the Profile-Guided Optimizer component described in the Murakkab system
(Section 3.3, Figure 6b). It transforms a logical workflow DAG (annotated with
candidate executors) into an *executable* workflow DAG by selecting concrete executors,
parameter configurations, and hardware assignments.

This MVP uses a **greedy heuristic strategy**:
- Selects the first available candidate executor for each task.
- Assigns default parameters and hardware (e.g., CPU).
- Annotates each node with estimated performance metrics (latency, cost, accuracy)
  based on available profile data.
- Performs minimal SLO validation.

For production, this component would evolve to solve a multi-objective optimization
problem balancing latency, accuracy, cost, and energy under SLO and resource constraints.
"""
# workflow_optimizer.py
from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx


class OptimizationError(RuntimeError):
    """Raised when workflow optimization fails or SLOs cannot be satisfied."""


class WorkflowOptimizer:
    """
    Profile- and resource-aware Workflow Optimizer.

    Given:
      - a logical DAG where each node has a 'candidates' list (Executor-like objects),
      - model_profiles mapping executor_id -> {latency_ms, accuracy, cost, device, ...},
      - hardware_resources describing available device types and capacities,

    Produces:
      - an executable DAG (networkx.DiGraph) annotated per-node with:
          * selected_executor (id)
          * executor_type
          * parameters (defaulted)
          * hardware (device, memory_gb)
          * estimated_latency_ms
          * estimated_cost
          * estimated_accuracy
          * optimization_score
      - graph["summary"] containing total metrics: critical path latency, total cost, min accuracy, meets_slos

    Notes:
      - This optimizer treats device counts as *concurrency capacity* and does not permanently
        consume a device slot for static selection. Instead, it verifies that each executor's
        memory requirement fits the target device capacity.
      - Critical path (max end time) is used for latency SLO checks.
    """

    def __init__(self, model_profiles: Dict[str, Dict[str, Any]], hardware_resources: Dict[str, Dict[str, Any]]):
        """
        Args:
            model_profiles: Mapping of executor_id -> profile dict with keys:
                - latency_ms (float)
                - accuracy (float in [0,1])
                - cost (float)
                - device (str) e.g., 'gpu' or 'cpu'
                (other keys allowed)
            hardware_resources: Mapping device_type -> resource dict, for example:
                {
                    "gpu": {"count": 2, "memory_gb": 24},
                    "cpu": {"count": 8, "memory_gb": 128}
                }
        """
        self.model_profiles = model_profiles or {}
        self.hardware_resources = hardware_resources or {}

    # ----------------------------
    # Public API
    # ----------------------------
    def optimize(self, logical_dag: nx.DiGraph, user_slos: Dict[str, Any]) -> nx.DiGraph:
        """
        Optimize the logical DAG into a concrete executable DAG.

        Args:
            logical_dag: networkx.DiGraph where each node attrs contain 'candidates' (list of Executor-like objects).
            user_slos: dictionary with optional keys: 'latency_ms', 'cost', 'accuracy'.

        Returns:
            networkx.DiGraph: executable DAG annotated as described above.

        Raises:
            OptimizationError: when no feasible plan can be found or SLOs cannot be satisfied.
        """
        if not isinstance(logical_dag, nx.DiGraph):
            raise TypeError("logical_dag must be a networkx.DiGraph")

        # Work on a shallow copy so original logical_dag is not mutated externally
        executable = logical_dag.copy()

        # Per-node selected metadata accumulator
        node_selection_info: Dict[str, Dict[str, Any]] = {}

        # For each node, pick the best feasible candidate according to scoring and hardware feasibility
        for node_name, attrs in executable.nodes(data=True):
            candidates = attrs.get("candidates", [])
            if not candidates:
                raise OptimizationError(f"Node '{node_name}' has no candidate executors.")

            best: Optional[Tuple[float, Dict[str, Any]]] = None  # (score, details)

            for cand in candidates:
                # Resolve executor id
                ex_id = getattr(cand, "id", None)
                if not ex_id:
                    # skip malformed candidate
                    continue

                profile = self.model_profiles.get(ex_id)
                if not profile:
                    # If no profile for this executor, skip candidate (cannot estimate)
                    continue

                # Extract profile fields with safe defaults
                latency = float(profile.get("latency_ms", 1000.0))
                accuracy = float(profile.get("accuracy", 1.0))
                cost = float(profile.get("cost", 0.0))
                device = profile.get("device", "cpu")

                # Resource requirements declared on executor (optional)
                resources = getattr(cand, "resources", {}) or {}
                mem_req = float(resources.get("memory_gb", 0.0))

                # Hardware feasibility: device type must exist and memory requirement must fit.
                hw_spec = self.hardware_resources.get(device)
                if hw_spec is None:
                    # device not available locally -> infeasible for static placement
                    continue
                hw_mem = float(hw_spec.get("memory_gb", 0.0))
                if mem_req > hw_mem:
                    # candidate requires more memory than any single device of this type provides -> infeasible
                    continue

                # Candidate is feasible from a hardware-capacity perspective.
                # Score candidate using normalized metrics.
                # We normalize latency and cost via a soft transformation so small differences matter.
                acc_norm = self._clamp(accuracy, 0.0, 1.0)
                # latency: convert to [0,1] preference where lower latency -> higher score
                lat_norm = 1.0 / (1.0 + latency)
                # cost: lower is better
                cost_norm = 1.0 / (1.0 + cost)

                # Weights: favor accuracy, then latency, then cost (tunable)
                w_acc, w_lat, w_cost = 0.5, 0.3, 0.2

                score = (w_acc * acc_norm) + (w_lat * lat_norm) + (w_cost * cost_norm)

                details = {
                    "executor_obj": cand,
                    "executor_id": ex_id,
                    "latency": latency,
                    "accuracy": accuracy,
                    "cost": cost,
                    "device": device,
                    "mem_req": mem_req,
                    "score": score,
                }

                if best is None or score > best[0]:
                    best = (score, details)

            if best is None:
                # Provide a helpful, actionable error message
                candidate_ids = [getattr(c, "id", "<no-id>") for c in candidates]
                available_devices = list(self.hardware_resources.keys())
                raise OptimizationError(
                    f"No feasible executor found for node '{node_name}'. "
                    f"Candidates considered: {candidate_ids}. "
                    f"Available device types: {available_devices}. "
                    "Likely reasons: missing model profile, required device not available, "
                    "or per-device memory insufficient."
                )

            # Store selection
            score, sel = best
            node_selection_info[node_name] = sel

        # Annotate executable DAG nodes with selection and default parameters
        for node_name, sel in node_selection_info.items():
            cand = sel["executor_obj"]
            parameters = self._default_parameters(cand)
            executable.nodes[node_name].update(
                {
                    "selected_executor": sel["executor_id"],
                    "executor_type": getattr(cand, "type", None),
                    "parameters": parameters,
                    "hardware": {"device": sel["device"], "memory_gb": sel["mem_req"]},
                    "estimated_latency_ms": sel["latency"],
                    "estimated_cost": sel["cost"],
                    "estimated_accuracy": sel["accuracy"],
                    "optimization_score": sel["score"],
                }
            )

        # Compute aggregate metrics:
        # - Critical path latency (longest path sum of node latencies)
        # - Total cost (sum)
        # - Min accuracy (min)
        total_cost = 0.0
        min_accuracy = 1.0
        for n, attrs in executable.nodes(data=True):
            total_cost += float(attrs.get("estimated_cost", 0.0))
            min_accuracy = min(min_accuracy, float(attrs.get("estimated_accuracy", 1.0)))

        critical_latency = self._compute_critical_path_latency(executable)

        # Validate SLOs
        slo_violations: List[str] = []
        if user_slos:
            if "latency_ms" in user_slos and critical_latency > float(user_slos["latency_ms"]):
                slo_violations.append(f"Critical-path latency {critical_latency:.1f}ms > SLO {user_slos['latency_ms']}ms")
            if "cost" in user_slos and total_cost > float(user_slos["cost"]):
                slo_violations.append(f"Total cost {total_cost:.3f} > SLO {user_slos['cost']}")
            if "accuracy" in user_slos and min_accuracy < float(user_slos["accuracy"]):
                slo_violations.append(f"Min accuracy {min_accuracy:.3f} < SLO {user_slos['accuracy']}")

            if slo_violations:
                # Provide summary and per-node metadata to help debugging
                summary_lines = ["SLO violations detected:"]
                summary_lines.extend(f" - {v}" for v in slo_violations)
                # Build a compact node summary
                node_summaries = []
                for n, a in executable.nodes(data=True):
                    node_summaries.append(
                        f"{n}: exec={a.get('selected_executor')}, lat={a.get('estimated_latency_ms')}, acc={a.get('estimated_accuracy')}, cost={a.get('estimated_cost')}"
                    )
                summary_lines.append("Node selections:")
                summary_lines.extend(f"   {s}" for s in node_summaries)
                raise OptimizationError("\n".join(summary_lines))

        # Annotate graph summary
        executable.graph["summary"] = {
            "critical_path_latency_ms": round(critical_latency, 2),
            "total_cost": round(total_cost, 4),
            "min_accuracy": round(min_accuracy, 4),
            "meets_slos": True,
        }

        return executable

    # ----------------------------
    # Helper utilities
    # ----------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _default_parameters(executor_obj: Any) -> Dict[str, Any]:
        """
        Instantiate conservative default parameter values for a candidate's parameter schema.
        Executor objects may or may not have a 'parameters' dict describing parameter types.
        """
        params = {}
        schema = getattr(executor_obj, "parameters", None)
        if not schema or not isinstance(schema, dict):
            return params

        for k, v in schema.items():
            t = (v or "").lower()
            if t == "int":
                params[k] = 1
            elif t == "float":
                params[k] = 0.0
            elif t == "str":
                params[k] = "default"
            else:
                params[k] = None
        return params

    @staticmethod
    def _compute_critical_path_latency(dag: nx.DiGraph) -> float:
        """
        Compute the critical path (longest-path) latency of a DAG where each node's
        weight is the node's estimated latency (ms). Returns the maximum end time.

        Approach:
            - Topologically order nodes.
            - For each node, earliest_start = max(earliest_finish of predecessors) (0 if none)
            - earliest_finish = earliest_start + node_latency
            - critical path = max(earliest_finish)
        """
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("DAG must be acyclic to compute critical path latency.")

        topo = list(nx.topological_sort(dag))
        earliest_finish: Dict[str, float] = {}

        for n in topo:
            latency = float(dag.nodes[n].get("estimated_latency_ms", 0.0))
            preds = list(dag.predecessors(n))
            if not preds:
                start = 0.0
            else:
                start = max(earliest_finish.get(p, 0.0) for p in preds)
            earliest_finish[n] = start + latency

        if not earliest_finish:
            return 0.0
        return max(earliest_finish.values())


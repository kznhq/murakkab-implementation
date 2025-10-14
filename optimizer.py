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

import networkx as nx
from typing import Dict, Any, List, Optional


class OptimizationError(Exception):
    """Raised when workflow optimization fails or SLOs cannot be satisfied."""


class WorkflowOptimizer:
    """
    Profile-Guided Workflow Optimizer (MVP implementation).

    Converts a logical DAG (with candidate executors) into a concrete executable
    workflow plan by selecting one executor per task and annotating estimated
    runtime metrics.
    """

    def __init__(self, model_profiles: Dict[str, Dict[str, Any]], hardware_resources: Dict[str, Any]):
        """
        Initialize the optimizer.

        Args:
            model_profiles: A dictionary of performance/accuracy/cost profiles keyed by executor ID.
                Example:
                {
                    "whisper-v3": {"latency_ms": 500, "accuracy": 0.93, "cost": 0.02, "device": "gpu"},
                    "frame-extractor-1": {"latency_ms": 150, "accuracy": 0.99, "cost": 0.001, "device": "cpu"}
                }

            hardware_resources: Dictionary describing available hardware/resource capacities.
                Example:
                {"cpu": {"count": 8, "memory_gb": 32}, "gpu": {"count": 1, "memory_gb": 24}}
        """
        self.model_profiles = model_profiles or {}
        self.hardware_resources = hardware_resources or {}

    def optimize(self, logical_dag: nx.DiGraph, user_slos: Dict[str, Any]) -> nx.DiGraph:
        """
        Optimize the logical workflow DAG into a concrete, executable DAG.

        Args:
            logical_dag: NetworkX DAG produced by `WorkflowOrchestrator.build_logical_plan()`.
                         Each node must have a `candidates` list of Executor objects.
            user_slos: Dictionary of service-level objectives, e.g.,
                       {"latency_ms": 1000, "accuracy": 0.9, "cost": 1.0}

        Returns:
            nx.DiGraph: Executable DAG annotated with:
                - selected_executor: Executor ID chosen
                - parameters: assigned parameter values (defaulted)
                - hardware: assigned hardware resource
                - estimated_latency_ms
                - estimated_cost
                - estimated_accuracy

        Raises:
            OptimizationError: if no feasible plan or SLO violation occurs.
        """
        if not isinstance(logical_dag, nx.DiGraph):
            raise TypeError("Expected a networkx.DiGraph from orchestrator")

        executable_dag = logical_dag.copy()

        total_latency = 0.0
        total_cost = 0.0
        min_accuracy = 1.0

        # Step 1: Select executor for each task node
        for node_name, attrs in executable_dag.nodes(data=True):
            candidates = attrs.get("candidates", [])
            if not candidates:
                raise OptimizationError(f"No candidate executors available for task '{node_name}'.")

            # Greedy heuristic: pick the first candidate
            selected_executor = candidates[0]

            # Look up profile data
            profile = self.model_profiles.get(selected_executor.id, {})
            latency = profile.get("latency_ms", 100)
            accuracy = profile.get("accuracy", 1.0)
            cost = profile.get("cost", 0.0)
            device = profile.get("device", "cpu")

            total_latency += latency
            total_cost += cost
            min_accuracy = min(min_accuracy, accuracy)

            # Assign default parameter values (if known)
            parameters = {}
            if hasattr(selected_executor, "parameters") and isinstance(selected_executor.parameters, dict):
                for p_name, p_type in selected_executor.parameters.items():
                    if p_type == "int":
                        parameters[p_name] = 1
                    elif p_type == "float":
                        parameters[p_name] = 0.0
                    elif p_type == "str":
                        parameters[p_name] = "default"
                    else:
                        parameters[p_name] = None

            # Annotate node
            executable_dag.nodes[node_name].update({
                "selected_executor": selected_executor.id,
                "executor_type": getattr(selected_executor, "type", None),
                "hardware": device,
                "parameters": parameters,
                "estimated_latency_ms": latency,
                "estimated_cost": cost,
                "estimated_accuracy": accuracy
            })

        # Step 2: Check SLOs
        if user_slos:
            if "latency_ms" in user_slos and total_latency > user_slos["latency_ms"]:
                raise OptimizationError(
                    f"Total estimated latency {total_latency:.1f}ms exceeds SLO limit {user_slos['latency_ms']}ms."
                )
            if "cost" in user_slos and total_cost > user_slos["cost"]:
                raise OptimizationError(
                    f"Total estimated cost {total_cost:.3f} exceeds SLO cost limit {user_slos['cost']}."
                )
            if "accuracy" in user_slos and min_accuracy < user_slos["accuracy"]:
                raise OptimizationError(
                    f"Minimum accuracy {min_accuracy:.3f} falls below required SLO {user_slos['accuracy']}."
                )

        # Step 3: Annotate overall workflow metrics
        executable_dag.graph["summary"] = {
            "total_latency_ms": total_latency,
            "total_cost": total_cost,
            "min_accuracy": min_accuracy,
            "meets_slos": True
        }

        return executable_dag


# --- Example Usage / Unit Test Harness ---
if __name__ == "__main__":
    from dataclasses import dataclass
    from workflow_orchestrator import WorkflowOrchestrator
    from workflow_orchestrator import ExecutorLibrary, Executor
    from workflow_dag_builder import build_workflow_dag

    # 1. Define example workflow spec
    example_workflow = {
        "nodes": [
            {"name": "scene_detect", "inputs": ["videos"], "outputs": ["scenes", "audio"]},
            {"name": "frame_extract", "inputs": ["scenes"], "outputs": ["frames"]},
            {"name": "speech_to_text", "inputs": ["audio"], "outputs": ["text"]}
        ],
        "edges": [
            {"from": "scene_detect", "to": "frame_extract", "data": "scenes"},
            {"from": "scene_detect", "to": "speech_to_text", "data": "audio"}
        ]
    }

    # 2. Register executors
    lib = ExecutorLibrary()
    lib.register_executor(Executor(id="scene-net-v1", type="mlmodel", inputs=["videos"], outputs=["scenes", "audio"]))
    lib.register_executor(Executor(id="frame-extractor-1", type="tool", inputs=["scenes"], outputs=["frames"]))
    lib.register_executor(Executor(id="whisper-v3", type="mlmodel", inputs=["audio"], outputs=["text"]))

    # 3. Build logical DAG via orchestrator
    orchestrator = WorkflowOrchestrator(lib)
    logical_plan = orchestrator.build_logical_plan(example_workflow)

    # 4. Define model profiles (mock data)
    model_profiles = {
        "scene-net-v1": {"latency_ms": 400, "accuracy": 0.92, "cost": 0.03, "device": "gpu"},
        "frame-extractor-1": {"latency_ms": 150, "accuracy": 0.99, "cost": 0.005, "device": "cpu"},
        "whisper-v3": {"latency_ms": 500, "accuracy": 0.95, "cost": 0.02, "device": "gpu"}
    }

    hardware_resources = {"cpu": {"count": 8}, "gpu": {"count": 1}}

    # 5. Optimize workflow
    optimizer = WorkflowOptimizer(model_profiles, hardware_resources)
    slos = {"latency_ms": 2000, "accuracy": 0.9, "cost": 0.1}

    executable_plan = optimizer.optimize(logical_plan, slos)

    # 6. Display final executable DAG
    print("\n=== Executable Workflow Plan ===")
    for n, data in executable_plan.nodes(data=True):
        print(f"Task: {n}")
        print(f"  Executor: {data['selected_executor']} ({data['executor_type']})")
        print(f"  Hardware: {data['hardware']}")
        print(f"  Latency: {data['estimated_latency_ms']} ms")
        print(f"  Accuracy: {data['estimated_accuracy']}")
        print(f"  Cost: {data['estimated_cost']}")
        print("  Params:", data.get("parameters", {}))
        print()

    print("=== Workflow Summary ===")
    print(executable_plan.graph["summary"])


"""
local_runtime_dispatcher.py

LocalRuntimeDispatcher

Executes an *executable* workflow DAG locally in topological order (with optional
parallel execution of independent branches). This runtime simulates executor
invocation (no external code is launched), logs execution, enforces dependencies,
and returns the outputs of terminal nodes.

Usage:
    dispatcher = LocalRuntimeDispatcher()
    final_outputs = dispatcher.execute(executable_dag, input_payload)

Author: ChatGPT (GPT-5 Thinking mini)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
from networkx.algorithms.dag import topological_generations


# ------------------------------------------------------------------------------
# Logging convenience: callers may configure logging as they wish. Provide a sane
# default if not configured.
# ------------------------------------------------------------------------------
DEFAULT_LOGGER_NAME = "LocalRuntimeDispatcher"


def _ensure_logger(name: str = DEFAULT_LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Basic console handler if none configured by the caller
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ------------------------------------------------------------------------------
# Exceptions used by dispatcher
# ------------------------------------------------------------------------------
class DispatchError(RuntimeError):
    """Base class for errors raised during workflow dispatch/execution."""


class MissingInputError(DispatchError):
    """Raised when a node requires an input that cannot be found in the data store."""


class ExecutionSimulationError(DispatchError):
    """Raised when a simulated executor indicates failure."""


# ------------------------------------------------------------------------------
# LocalRuntimeDispatcher
# ------------------------------------------------------------------------------
class LocalRuntimeDispatcher:
    """
    Executes an executable workflow DAG locally in topological order.

    The dispatcher expects the DAG nodes to be "executable" (i.e., an optimizer has
    selected a concrete executor for each node). Each node should contain at least:

        - 'selected_executor': str (executor id)
        - 'parameters': dict (executor parameters)
        - 'hardware': dict (assigned hardware resources)
        - 'estimated_latency_ms': float (used to simulate delay)
        - 'estimated_cost': float (informational)
        - 'estimated_accuracy': float (informational)
        - 'inputs': List[str] (logical input names)
        - 'outputs': List[str] (logical output names)

    The dispatcher:
        - respects dependencies (predecessors must complete first),
        - collects required inputs from prior node outputs or initial payload,
        - simulates executor runs with time.sleep(), logs progress,
        - supports parallel execution of nodes in the same topological generation,
        - returns a dict of outputs from terminal node(s).

    Note: This dispatcher is intentionally request-agnostic and does not run real
    executors or assign resources. It is an execution *simulator/MVP*.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Args:
            logger: Optional pre-configured logger. If None a default logger is created.
        """
        self.logger = logger or _ensure_logger()
        # internal stores reset per execution
        self._execution_results: Dict[str, Dict[str, Any]] = {}
        self._data_store: Dict[str, Any] = {}

    # --------------------------
    # Public API
    # --------------------------
    def execute(self, dag: nx.DiGraph, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given executable DAG locally (simulation).

        Args:
            dag: networkx.DiGraph annotated with executable node attributes.
            input_payload: initial inputs mapping data-name -> value.

        Returns:
            A dict combining outputs of terminal nodes (nodes with out-degree == 0).

        Raises:
            DispatchError subclasses on problems (missing inputs, execution failure).
        """
        # Basic validation of DAG type
        if not isinstance(dag, nx.DiGraph):
            raise DispatchError("dag must be a networkx.DiGraph")

        self.logger.info("Starting workflow execution")

        # Reset per-execution stores
        self._execution_results.clear()
        self._data_store.clear()
        # Seed data store with initial payload
        self._data_store.update(input_payload or {})

        # Execute nodes generation-by-generation (parallelize within a generation)
        try:
            for gen_idx, generation in enumerate(topological_generations(dag)):
                nodes = list(generation)
                self.logger.debug(f"Topological generation {gen_idx}: {nodes}")

                # Use parallel execution for nodes within the same generation.
                # These nodes are independent with respect to each other (no inter-edges).
                if len(nodes) == 1:
                    # Single node: run inline to make logs straightforward
                    node = nodes[0]
                    self._run_node(dag, node)
                else:
                    # Parallel execution
                    with ThreadPoolExecutor(max_workers=min(8, len(nodes))) as exe:
                        futures = {exe.submit(self._run_node, dag, node): node for node in nodes}
                        for fut in as_completed(futures):
                            node = futures[fut]
                            try:
                                fut.result()
                            except Exception as exc:
                                # On failure, log and raise to terminate the workflow
                                self.logger.error(f"Node '{node}' failed: {exc}")
                                raise

            self.logger.info("Workflow execution completed")
        except Exception:
            self.logger.exception("Workflow execution terminated due to error")
            raise

        # Aggregate terminal outputs
        terminal_nodes = [n for n in dag.nodes if dag.out_degree(n) == 0]
        final_outputs: Dict[str, Any] = {}
        for tn in terminal_nodes:
            res = self._execution_results.get(tn, {})
            # If node declares named outputs, include them keyed by those names
            outputs = dag.nodes[tn].get("outputs", []) or list(res.keys())
            if outputs:
                for out_name in outputs:
                    # Prefer explicit produced data if available in data_store
                    if out_name in self._data_store:
                        final_outputs[out_name] = self._data_store[out_name]
                    else:
                        # fallback to node-scoped result mapping
                        final_outputs[out_name] = res.get(out_name, None)
            else:
                # If no declared outputs, place the whole node result under node name
                final_outputs[tn] = res

        return final_outputs

    # --------------------------
    # Internal helpers
    # --------------------------
    def _run_node(self, dag: nx.DiGraph, node: str) -> None:
        """
        Execute a single node (simulation). Collects inputs, logs, sleeps to
        simulate latency, stores outputs.

        This function is thread-safe as long as each generation's nodes write to
        disjoint output keys (which is true if the DAG is well-formed) and
        previous generations have completed before this generation starts.
        """
        node_attrs = dag.nodes[node]
        sel_exec = node_attrs.get("selected_executor")
        parameters = node_attrs.get("parameters", {}) or {}
        hardware = node_attrs.get("hardware", {}) or {}
        est_latency_ms = float(node_attrs.get("estimated_latency_ms", 100.0) or 100.0)
        est_cost = node_attrs.get("estimated_cost")
        est_accuracy = node_attrs.get("estimated_accuracy")

        # Resolve inputs for this node:
        required_inputs: List[str] = node_attrs.get("inputs", []) or []
        input_values: Dict[str, Any] = {}
        missing_inputs: List[str] = []

        for inp in required_inputs:
            if inp in self._data_store:
                input_values[inp] = self._data_store[inp]
            else:
                missing_inputs.append(inp)

        if missing_inputs:
            msg = (
                f"Node '{node}' missing required inputs: {missing_inputs}. "
                f"Available data store keys: {sorted(list(self._data_store.keys()))}"
            )
            self.logger.error(msg)
            raise MissingInputError(msg)

        # Log the start of execution
        self.logger.info(f"Executing node '{node}' using executor '{sel_exec}'")
        self.logger.debug(f"  Inputs: {input_values}")
        self.logger.debug(f"  Parameters: {parameters}")
        self.logger.debug(f"  Hardware: {hardware}")
        self.logger.debug(f"  Estimated latency (ms): {est_latency_ms}")
        if est_cost is not None:
            self.logger.debug(f"  Estimated cost: {est_cost}")
        if est_accuracy is not None:
            self.logger.debug(f"  Estimated accuracy: {est_accuracy}")

        # Simulate execution - allow a special parameter to force a simulated failure
        try:
            # Simulated pre-execution delay (very small to show concurrency if used)
            sleep_s = max(0.0, est_latency_ms / 1000.0)
            time.sleep(sleep_s)

            # Simulate a failure if explicitly requested in parameters (useful for tests)
            if parameters.get("simulate_failure") is True:
                raise ExecutionSimulationError(f"Simulated failure requested for node '{node}'")

            # Build simulated outputs. Use declared outputs when available.
            outputs_decl = node_attrs.get("outputs", [])
            node_result: Dict[str, Any] = {}
            if outputs_decl:
                for out_name in outputs_decl:
                    simulated_value = f"simulated_result_from_{node}_{out_name}"
                    node_result[out_name] = simulated_value
                    # Update shared data store so downstream nodes can consume outputs by name
                    self._data_store[out_name] = simulated_value
            else:
                # If no declared outputs, produce a generic node-scoped output
                node_result[f"{node}_output"] = f"simulated_result_from_{node}"
                # Store under the node key too
                self._data_store[node] = node_result[f"{node}_output"]

            # Save node-level execution result
            self._execution_results[node] = node_result

            self.logger.info(f"Node '{node}' completed successfully")
            self.logger.debug(f"  Output -> {node_result}")
        except Exception as exc:
            # Log and re-raise for upstream handling
            self.logger.error(f"Execution of node '{node}' failed: {exc}")
            raise

# ------------------------------------------------------------------------------
# Example test script (runs when module is executed directly)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Configure logging for demo
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate handlers in environments that
    # pre-configure logging (like some REPLs).
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        stream=sys.stdout,
    )

    # Build a sample executable DAG
    G = nx.DiGraph()
    # Node: scene_detect (source)
    G.add_node(
        "scene_detect",
        inputs=["videos"],
        outputs=["scenes", "audio"],
        selected_executor="scene-net-v1",
        parameters={"sensitivity": "medium"},
        hardware={"device": "cpu", "memory_gb": 4},
        estimated_latency_ms=150.0,
        estimated_cost=0.02,
        estimated_accuracy=0.9,
    )
    # Node: frame_extract (depends on scenes)
    G.add_node(
        "frame_extract",
        inputs=["scenes"],
        outputs=["frames"],
        selected_executor="frame-extractor-1",
        parameters={},
        hardware={"device": "cpu", "memory_gb": 2},
        estimated_latency_ms=80.0,
        estimated_cost=0.005,
        estimated_accuracy=0.99,
    )
    # Node: speech_to_text (depends on audio)
    G.add_node(
        "speech_to_text",
        inputs=["audio"],
        outputs=["transcript"],
        selected_executor="whisper-v3",
        parameters={"language": "en", "beam_size": 5},
        hardware={"device": "gpu", "memory_gb": 12},
        estimated_latency_ms=250.0,
        estimated_cost=0.10,
        estimated_accuracy=0.92,
    )
    # Node: summary (depends on frames and transcript) - demonstrates a join
    G.add_node(
        "summarize",
        inputs=["frames", "transcript"],
        outputs=["summary"],
        selected_executor="llama-3.2",
        parameters={"max_tokens": 128},
        hardware={"device": "gpu", "memory_gb": 16},
        estimated_latency_ms=300.0,
        estimated_cost=0.5,
        estimated_accuracy=0.88,
    )

    # Edges (dataflow)
    G.add_edge("scene_detect", "frame_extract", data="scenes")
    G.add_edge("scene_detect", "speech_to_text", data="audio")
    G.add_edge("frame_extract", "summarize", data="frames")
    G.add_edge("speech_to_text", "summarize", data="transcript")

    # Basic DAG validation
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Sample DAG contains a cycle, aborting demo.")

    # Initial inputs
    initial_inputs = {"videos": "roadtrip.mp4"}

    # Create dispatcher and run
    dispatcher = LocalRuntimeDispatcher()
    final = dispatcher.execute(G, initial_inputs)

    print("\nFinal workflow outputs:")
    for k, v in final.items():
        print(f" - {k}: {v}")


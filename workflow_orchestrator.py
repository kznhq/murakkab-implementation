"""
Workflow Orchestrator

Transforms a parsed workflow DAG (from `build_workflow_dag`) and an ExecutorLibrary
into an annotated, request-agnostic logical workflow plan.

The orchestrator:
- For each task node, discovers candidate executors that can implement the task.
- Validates data-type compatibility along each data-flow edge.
- Annotates nodes with `candidates` (list of Executor objects).
- Raises descriptive errors if tasks are unassignable or edges are incompatible.

This module expects an external `build_workflow_dag(json_spec) -> networkx.DiGraph`
function and an `ExecutorLibrary` object with at least the following methods:

- query_executors(required_inputs: List[str], required_outputs: List[str]) -> List[Executor]
- get_executor_by_id(executor_id: str) -> Executor  # optional; not required here

An `Executor` is expected to have attributes:
- id: str
- type: str
- inputs: List[str]
- outputs: List[str]

"""

from dataclasses import dataclass
from typing import List, Any, Dict
import networkx as nx
from executor_library import *
from workflow_dag_builder import *

# ---- Exceptions ----
class OrchestratorError(Exception):
    """Base class for orchestrator errors."""


class UnassignableTaskError(OrchestratorError):
    """Raised when no executor candidates are available for a task."""


class TypeMismatchError(OrchestratorError):
    """Raised when data-flow types on an edge are incompatible."""


# ---- Workflow Orchestrator ----
class WorkflowOrchestrator:
    """
    Maps tasks in a workflow DAG to candidate executors and validates type compatibility.

    The orchestrator is request-agnostic: it performs static analysis and annotation,
    not runtime/resource assignment.
    """

    def __init__(self, executor_lib: ExecutorLibrary):
        """
        Args:
            executor_lib: An ExecutorLibrary instance used to discover candidate executors.
        """
        self.executor_lib = executor_lib

    def _node_required_ios(self, node_attrs: Dict[str, Any]) -> (List[str], List[str]):
        """
        Normalize and return (inputs, outputs) lists for a node.
        """
        inputs = node_attrs.get("inputs", []) or []
        outputs = node_attrs.get("outputs", []) or []
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise OrchestratorError("Node 'inputs' and 'outputs' must be lists.")
        return inputs, outputs

    def _filter_executors_strict(
        self, candidates: List[Executor], required_inputs: List[str], required_outputs: List[str]
    ) -> List[Executor]:
        """
        From a candidate list, keep only executors that strictly provide at least all
        required inputs and outputs.

        An executor is considered compatible if:
            set(required_inputs) subset-of set(executor.inputs)
            AND set(required_outputs) subset-of set(executor.outputs)
        """
        req_in_set = set(required_inputs)
        req_out_set = set(required_outputs)
        filtered = []
        for ex in candidates:
            # Defensive attribute access: assume executor-like object
            ex_inputs = set(getattr(ex, "inputs", []) or [])
            ex_outputs = set(getattr(ex, "outputs", []) or [])
            if req_in_set.issubset(ex_inputs) and req_out_set.issubset(ex_outputs):
                filtered.append(ex)
        return filtered

    def build_logical_plan(self, workflow_spec_json: dict) -> nx.DiGraph:
        """
        Build an annotated logical workflow DAG.

        Args:
            workflow_spec_json: Declarative workflow JSON/dict (nodes + edges).
            build_workflow_dag: callable(json_spec) -> nx.DiGraph that parses the JSON
                               and returns a directed graph where node keys are task names
                               and node attributes include 'inputs' and 'outputs'.

        Returns:
            networkx.DiGraph: The same DAG structure but each node is annotated with
                              a `candidates` attribute (list of Executor objects).

        Raises:
            UnassignableTaskError: if a task cannot be matched to any executor.
            TypeMismatchError: if an edge references data not produced/consumed or
                               there are no compatible executors that produce/consume it.
            OrchestratorError: for other unexpected validation problems.
        """ 
        # Step 1: parse into a task DAG using provided builder
        dag = build_workflow_dag(workflow_spec_json)
        if not isinstance(dag, nx.DiGraph):
            raise OrchestratorError("build_workflow_dag must return a networkx.DiGraph")

        # Step 2: Annotate each node with candidate executors
        for node_name, attrs in list(dag.nodes(data=True)):
            inputs, outputs = self._node_required_ios(attrs)

            # Ask library for candidate executors (library may return superset)
            try:
                candidates_raw = self.executor_lib.query_executors(input_type=inputs, output_type=outputs)
            except TypeError:
                # in case the library expects positional args or different signature, try a relaxed call
                candidates_raw = self.executor_lib.query_executors(inputs, outputs)  # type: ignore

            # Filter strictly to executors that actually cover required IO
            candidates = self._filter_executors_strict(candidates_raw, inputs, outputs)

            if not candidates:
                # Build a helpful message including what the library returned (if any)
                returned_ids = [getattr(c, "id", "<unknown>") for c in candidates_raw] if candidates_raw else []
                raise UnassignableTaskError(
                    f"Task '{node_name}' (inputs={inputs}, outputs={outputs}) cannot be assigned to any executor. "
                    f"Library returned candidates: {returned_ids}."
                )

            # Attach candidates list to node attributes
            dag.nodes[node_name]["candidates"] = candidates

        # Step 3: Type-check edges
        # Edges are expected to have attribute 'data' naming the produced data item (a string or list)
        for src, dst, edge_attrs in list(dag.edges(data=True)):
            data_item = edge_attrs.get("data")
            if data_item is None:
                raise TypeMismatchError(f"Edge {src} -> {dst} is missing required 'data' attribute.")

            # Normalize to single string (spec expects a single data string per edge)
            if isinstance(data_item, list):
                # For simplicity we require edges that carry multiple data items to list them individually.
                if len(data_item) != 1:
                    # if multiple, validate each individually
                    raise TypeMismatchError(
                        f"Edge {src} -> {dst} specifies multiple data items {data_item}; "
                        f"the orchestrator expects single-edge data labels. Split edges in the workflow spec."
                    )
                data_item = data_item[0]

            # Check declared capabilities at node level (original node attributes)
            src_outputs = set(dag.nodes[src].get("outputs", []))
            dst_inputs = set(dag.nodes[dst].get("inputs", []))

            if data_item not in src_outputs:
                raise TypeMismatchError(
                    f"Edge {src} -> {dst} declares data '{data_item}' but source task '{src}' "
                    f"does not declare that output (declared outputs: {sorted(src_outputs)})."
                )
            if data_item not in dst_inputs:
                raise TypeMismatchError(
                    f"Edge {src} -> {dst} declares data '{data_item}' but destination task '{dst}' "
                    f"does not declare that input (declared inputs: {sorted(dst_inputs)})."
                )

            # Now ensure at least one candidate executor for src produces data_item
            src_candidates = dag.nodes[src]["candidates"]
            dst_candidates = dag.nodes[dst]["candidates"]

            producers = [ex for ex in src_candidates if data_item in getattr(ex, "outputs", [])]
            consumers = [ex for ex in dst_candidates if data_item in getattr(ex, "inputs", [])]

            if not producers:
                raise TypeMismatchError(
                    f"No candidate executor for source task '{src}' produces data item '{data_item}'. "
                    f"Source candidates: {[getattr(e,'id',None) for e in src_candidates]}"
                )
            if not consumers:
                raise TypeMismatchError(
                    f"No candidate executor for destination task '{dst}' consumes data item '{data_item}'. "
                    f"Destination candidates: {[getattr(e,'id',None) for e in dst_candidates]}"
                )

            # Optionally annotate the edge with metadata about producer/consumer candidate IDs
            dag.edges[src, dst].setdefault("producers", [getattr(p, "id", None) for p in producers])
            dag.edges[src, dst].setdefault("consumers", [getattr(c, "id", None) for c in consumers])

        # All checks passed; return annotated DAG
        return dag


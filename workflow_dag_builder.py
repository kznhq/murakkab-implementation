"""
workflow_dag_builder.py

This module provides functionality to parse a declarative JSON workflow specification
and construct a Directed Acyclic Graph (DAG) representation using NetworkX.

Example:
    >>> import json
    >>> from workflow_dag_builder import build_workflow_dag
    >>> workflow_json = {
    ...     "nodes": [
    ...         {"name": "scene_detect", "inputs": ["videos"], "outputs": ["scenes", "audio"]},
    ...         {"name": "frame_extract", "inputs": ["scenes"], "outputs": ["frames"]},
    ...         {"name": "speech_to_text", "inputs": ["audio"], "outputs": ["transcript"]}
    ...     ],
    ...     "edges": [
    ...         {"from": "scene_detect", "to": "frame_extract", "data": "scenes"},
    ...         {"from": "scene_detect", "to": "speech_to_text", "data": "audio"}
    ...     ]
    ... }
    >>> dag = build_workflow_dag(workflow_json)
    >>> list(dag.nodes)
    ['scene_detect', 'frame_extract', 'speech_to_text']
"""

import networkx as nx


class WorkflowValidationError(Exception):
    """Raised when the workflow specification is invalid."""
    pass


def validate_workflow_spec(json_workflow: dict) -> None:
    """
    Validate the workflow specification for structure and consistency.

    Args:
        json_workflow (dict): Declarative workflow specification.

    Raises:
        WorkflowValidationError: If validation fails due to missing fields,
                                 duplicate task names, or invalid references.
    """
    if not isinstance(json_workflow, dict):
        raise WorkflowValidationError("Workflow must be a dictionary.")

    # Validate presence of keys
    if "nodes" not in json_workflow or "edges" not in json_workflow:
        raise WorkflowValidationError("Workflow must contain 'nodes' and 'edges' keys.")

    nodes = json_workflow["nodes"]
    edges = json_workflow["edges"]

    # Validate node structure
    if not isinstance(nodes, list):
        raise WorkflowValidationError("'nodes' must be a list.")
    if not all(isinstance(n, dict) and "name" in n for n in nodes):
        raise WorkflowValidationError("Each node must be a dict with a 'name' field.")

    # Ensure unique node names
    node_names = [n["name"] for n in nodes]
    if len(set(node_names)) != len(node_names):
        raise WorkflowValidationError("Duplicate task names detected in nodes.")

    # Validate edges
    if not isinstance(edges, list):
        raise WorkflowValidationError("'edges' must be a list.")

    for edge in edges:
        if not all(k in edge for k in ("from", "to", "data")):
            raise WorkflowValidationError("Each edge must contain 'from', 'to', and 'data' fields.")
        if edge["from"] not in node_names:
            raise WorkflowValidationError(f"Edge 'from' task '{edge['from']}' not found in nodes.")
        if edge["to"] not in node_names:
            raise WorkflowValidationError(f"Edge 'to' task '{edge['to']}' not found in nodes.")


def build_workflow_dag(json_workflow: dict) -> nx.DiGraph:
    """
    Construct a NetworkX Directed Acyclic Graph (DAG) from a workflow JSON specification.

    Args:
        json_workflow (dict): Workflow specification containing nodes and edges.

    Returns:
        nx.DiGraph: A directed acyclic graph representing the workflow.

    Raises:
        WorkflowValidationError: If validation fails or a cycle is detected.
    """
    validate_workflow_spec(json_workflow)

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in json_workflow["nodes"]:
        name = node["name"]
        inputs = node.get("inputs", [])
        outputs = node.get("outputs", [])
        G.add_node(name, inputs=inputs, outputs=outputs)

    # Add edges representing data dependencies
    for edge in json_workflow["edges"]:
        src = edge["from"]
        dst = edge["to"]
        data = edge["data"]
        G.add_edge(src, dst, data=data)

    # Check for cycles
    if not nx.is_directed_acyclic_graph(G):
        raise WorkflowValidationError("Cycle detected in workflow DAG.")

    return G


if __name__ == "__main__":
    # Example usage
    import json

    example_workflow = {
        "nodes": [
            {"name": "scene_detect", "inputs": ["videos"], "outputs": ["scenes", "audio"]},
            {"name": "frame_extract", "inputs": ["scenes"], "outputs": ["frames"]},
            {"name": "speech_to_text", "inputs": ["audio"], "outputs": ["transcript"]}
        ],
        "edges": [
            {"from": "scene_detect", "to": "frame_extract", "data": "scenes"},
            {"from": "scene_detect", "to": "speech_to_text", "data": "audio"}
        ]
    }

    dag = build_workflow_dag(example_workflow)
    print("Workflow DAG successfully built:")
    print("Nodes:", dag.nodes(data=True))
    print("Edges:", list(dag.edges(data=True)))


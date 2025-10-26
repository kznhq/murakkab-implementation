import sys
import networkx as nx
import logging
from workflow_dag_builder import *
from executor_library import *
from workflow_orchestrator import *
from optimizer import *
from dispatcher import *

if __name__ == '__main__':
    # example workflow from paper
    video_qa_workflow = {
        "nodes": [
            {"name": "scene_detector", "inputs": ["video"], "outputs": ["scenes", "audio"]},
            {"name": "frame_extractor", "inputs": ["scenes"], "outputs": ["frames"]},
            {"name": "speech_to_text", "inputs": ["audio"], "outputs": ["transcript"]},
            {"name": "object_detector", "inputs": ["frames"], "outputs": ["objframes"]},
            {"name": "question_answer", "inputs": ["query", "transcript", "objframes"], "outputs": ["answer"]}
        ],
        "edges": [
            {"from": "scene_detector", "to": "frame_extractor", "data": "scenes"},
            {"from": "scene_detector", "to": "speech_to_text", "data": "audio"},
            {"from": "frame_extractor", "to": "object_detector", "data": "frames"},
            {"from": "speech_to_text", "to": "question_answer", "data": "transcript"},
            {"from": "object_detector", "to": "question_answer", "data": "objframes"}
        ]
    }

    # build the generic DAG from the workflow (this is not a step in Murakkab, just a helper function so here I'm just calling it so we can see the generic DAG)
    dag = build_workflow_dag(video_qa_workflow)
    print("Workflow DAG successfully built:")
    print("Nodes:")
    nodes = dag.nodes(data=True)
    for node in nodes:
        print(node)
    print("Edges:")
    edges = list(dag.edges(data=True))
    for edge in edges:
        print(edge)

    # register some executors
    exec_lib = ExecutorLibrary()

    whisper_executor = Executor(
        id='whisper-v3',
        type='mlmodel',
        inputs=['audio'],
        outputs=['transcript'],
        parameters={'language': 'str', 'beam_size': 'int'},
        # resources={'device': 'gpu', 'memory_gb': '12'} # this is the minimum requirements to run the executor, not the profiling (profiling is separate as shown later). Also these numbers are arbitrary right now for sake of demo
    )
    opencv_executor = Executor(
        id='opencv-scene-detect',
        type='tool',
        inputs=['video'],
        outputs=['scenes', 'audio']
    )
    frame_extractor_executor = Executor(
        id='frame-extractor',
        type='tool',
        inputs=['scenes'],
        outputs=['frames']
    )
    object_detector_executor = Executor(
        id='yolov8-object-detect',
        type='mlmodel',
        inputs=['frames'],
        outputs=['objframes']
    )
    llama_executor = Executor(
        id='llama-3.2',
        type='llm',
        inputs=['query', 'transcript', 'objframes'],
        outputs=['answer']
    )
    gpt_executor = Executor(
        id='gpt-4o',
        type='llm',
        inputs=['query', 'transcript', 'objframes'],
        outputs=['answer']
    )
    exec_lib.register_executor(whisper_executor)
    exec_lib.register_executor(opencv_executor)
    exec_lib.register_executor(frame_extractor_executor)
    exec_lib.register_executor(object_detector_executor)
    exec_lib.register_executor(llama_executor)
    exec_lib.register_executor(gpt_executor)

    # all_executors = exec_lib.list_all_executors()
    # print(f"All executors: {[e.id for e in all_executors]}")

    # separator line
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Workflow Orchestrator takes generic DAG and assigns tools from executor library
    orchestrator = WorkflowOrchestrator(exec_lib)
    logical_dag = orchestrator.build_logical_plan(video_qa_workflow)
    print("Logical DAG successfully built:")
    print("Nodes:")
    nodes = logical_dag.nodes(data=True)
    for node in nodes:
        print(node)
    print("Edges:")
    edges = list(logical_dag.edges(data=True))
    for edge in edges:
        print(edge)

    # separator line
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Optimizer will choose best executors given our constraints

    # profiles of each executor kept here instead of library (numbers here just made up for sake of example)
    model_profiles = {
        "whisper-v3": {"latency_ms": 400, "accuracy": 0.94, "cost": 0.03, "device": "gpu"},
        "opencv-scene-detect": {"latency_ms": 800, "accuracy": 0.91, "cost": 0.05, "device": "gpu"},
        "frame-extractor": {"latency_ms": 700, "accuracy": 0.97, "cost": 0.04, "device": "gpu"},
        "yolov8-object-detect": {"latency_ms": 700, "accuracy": 0.93, "cost": 0.04, "device": "gpu"},
        "llama-3.2": {"latency_ms": 1200, "accuracy": 0.85, "cost": 0.11, "device": "gpu"},
        "gpt-4o": {"latency_ms": 900, "accuracy": 0.98, "cost": 0.09, "device": "gpu"}
    }

    # Our hardware constraints (numbers here made up for sake of example)
    hardware_resources = {"cpu": {"count": 8}, "gpu": {"count": 1}}

    # Optimize workflow (slos numbers are just made up for sake of example)
    optimizer = WorkflowOptimizer(model_profiles, hardware_resources)
    slos = {"latency_ms": 4000, "accuracy": 0.9, "cost": 0.3}

    executable_dag = optimizer.optimize(logical_dag, slos)

    # Display final executable DAG
    print("\n=== Executable Workflow Plan ===")
    for n, data in executable_dag.nodes(data=True):
        print(f"Task: {n}")
        print(f"  Executor: {data['selected_executor']} ({data['executor_type']})")
        print(f"  Hardware: {data['hardware']}")
        print(f"  Latency: {data['estimated_latency_ms']} ms")
        print(f"  Accuracy: {data['estimated_accuracy']}")
        print(f"  Cost: {data['estimated_cost']}")
        print("  Params:", data.get("parameters", {}))
        print()

    print("=== Workflow Summary ===")
    print(executable_dag.graph["summary"])

    # separator line
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Dispatcher to actually run the DAG

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

    # Basic DAG validation
    if not nx.is_directed_acyclic_graph(executable_dag):
        raise RuntimeError("Produced DAG contains a cycle, aborting.")

    # Initial inputs to run the workflow on
    initial_inputs = {"video": "roadtrip.mp4", "query": "Who is the person in the red dress?"}

    # Create dispatcher and run
    dispatcher = Dispatcher()
    final = dispatcher.execute(executable_dag, initial_inputs)

    print("\nFinal workflow outputs:")
    for k, v in final.items():
        print(f" - {k}: {v}")


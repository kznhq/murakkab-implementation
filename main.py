from workflow_dag_builder import *
from executor_library import *
from workflow_orchestrator import *

if __name__ == '__main__':
    # example workflow
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

    # build the generic DAG from the workflow
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
        resources={'device': 'gpu', 'memory_gb': '12'}
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
    exec_lib.register_executor(whisper_executor)
    exec_lib.register_executor(opencv_executor)
    exec_lib.register_executor(frame_extractor_executor)
    exec_lib.register_executor(object_detector_executor)
    exec_lib.register_executor(llama_executor)

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

# Murakkab Implementation
Implementing Murakkab from the [paper](https://arxiv.org/pdf/2508.18298)

## Architecture/Sequence of events
1. workflow (JSON)
2. define executor_library.Executor's and a executor_library.ExecutorLibrary() and use executor_library.register_executor(Executor) to register the Executors
    - doesn't have to be after building generic DAG, could be before step 2 or 1 too
3. define a workflow_orchestrator.WorkflowOrchestrator(ExecutorLibrary) then WorkflowOrchestrator.build_logical_plan(workflow)
    - this builds a DAG with nodes for each task in workflow and candidates for each node
4. define some profiles, resources, SLOs then use workflow_optimizer.WorkflowOptimizer(profiles: dict, resources: dict) and WorkflowOptimizer.optimize(logical DAG, SLOs)
    - examples in example.py but basically a bunch of dicts defining profiles of the executors, the resources, and the SLOs
    - optimize() will select one of the candidates to use for actual execution
    - TODO: currently just does greedy approach of using first candidate listed, need to flesh it out to actually optimize
5. dispatcher.execute(DAG from optimize() step, initial inputs: JSON) then answer is printed
    - currently simulates tool calls
    - TODO: make it actually run tools

## To run:
example.py goes through an example of how to setup workflow, initial inputs, etc.
```
pip install -r requirements.txt
python3 example.py
```
Note: The pip install is only to make sure you have the proper packages.

## Other TODOs maybe:
- make REST endpoints or equivalent so that you can register workflows and execute workflows with different calls
- scaling?
- dynamic profiling that updates the profiles used by optimizer
    - also just measuring performance/profiling in general

## Output from example.py
```
Workflow DAG successfully built:
Nodes:
('scene_detector', {'inputs': ['video'], 'outputs': ['scenes', 'audio']})
('frame_extractor', {'inputs': ['scenes'], 'outputs': ['frames']})
('speech_to_text', {'inputs': ['audio'], 'outputs': ['transcript']})
('object_detector', {'inputs': ['frames'], 'outputs': ['objframes']})
('question_answer', {'inputs': ['query', 'transcript', 'objframes'], 'outputs': ['answer']})
Edges:
('scene_detector', 'frame_extractor', {'data': 'scenes'})
('scene_detector', 'speech_to_text', {'data': 'audio'})
('frame_extractor', 'object_detector', {'data': 'frames'})
('speech_to_text', 'question_answer', {'data': 'transcript'})
('object_detector', 'question_answer', {'data': 'objframes'})
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logical DAG successfully built:
Nodes:
('scene_detector', {'inputs': ['video'], 'outputs': ['scenes', 'audio'], 'candidates': [Executor(id='opencv-scene-detect', type='tool', inputs=['video'], outputs=['scenes', 'audio'], parameters={}, resources={})]})
('frame_extractor', {'inputs': ['scenes'], 'outputs': ['frames'], 'candidates': [Executor(id='frame-extractor', type='tool', inputs=['scenes'], outputs=['frames'], parameters={}, resources={})]})
('speech_to_text', {'inputs': ['audio'], 'outputs': ['transcript'], 'candidates': [Executor(id='whisper-v3', type='mlmodel', inputs=['audio'], outputs=['transcript'], parameters={'language': 'str', 'beam_size': 'int'}, resources={'device': 'gpu', 'memory_gb': '12'})]})
('object_detector', {'inputs': ['frames'], 'outputs': ['objframes'], 'candidates': [Executor(id='yolov8-object-detect', type='mlmodel', inputs=['frames'], outputs=['objframes'], parameters={}, resources={})]})
('question_answer', {'inputs': ['query', 'transcript', 'objframes'], 'outputs': ['answer'], 'candidates': [Executor(id='llama-3.2', type='llm', inputs=['query', 'transcript', 'objframes'], outputs=['answer'], parameters={}, resources={})]})
Edges:
('scene_detector', 'frame_extractor', {'data': 'scenes', 'producers': ['opencv-scene-detect'], 'consumers': ['frame-extractor']})
('scene_detector', 'speech_to_text', {'data': 'audio', 'producers': ['opencv-scene-detect'], 'consumers': ['whisper-v3']})
('frame_extractor', 'object_detector', {'data': 'frames', 'producers': ['frame-extractor'], 'consumers': ['yolov8-object-detect']})
('speech_to_text', 'question_answer', {'data': 'transcript', 'producers': ['whisper-v3'], 'consumers': ['llama-3.2']})
('object_detector', 'question_answer', {'data': 'objframes', 'producers': ['yolov8-object-detect'], 'consumers': ['llama-3.2']})
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=== Executable Workflow Plan ===
Task: scene_detector
  Executor: opencv-scene-detect (tool)
  Hardware: gpu
  Latency: 800 ms
  Accuracy: 0.91
  Cost: 0.05
  Params: {}

Task: frame_extractor
  Executor: frame-extractor (tool)
  Hardware: gpu
  Latency: 700 ms
  Accuracy: 0.97
  Cost: 0.04
  Params: {}

Task: speech_to_text
  Executor: whisper-v3 (mlmodel)
  Hardware: gpu
  Latency: 400 ms
  Accuracy: 0.94
  Cost: 0.03
  Params: {'language': 'default', 'beam_size': 1}

Task: object_detector
  Executor: yolov8-object-detect (mlmodel)
  Hardware: gpu
  Latency: 700 ms
  Accuracy: 0.93
  Cost: 0.04
  Params: {}

Task: question_answer
  Executor: llama-3.2 (llm)
  Hardware: gpu
  Latency: 900 ms
  Accuracy: 0.98
  Cost: 0.09
  Params: {}

=== Workflow Summary ===
{'total_latency_ms': 3500.0, 'total_cost': 0.25, 'min_accuracy': 0.91, 'meets_slos': True}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[INFO] 2025-10-15 02:17:51,021 - Dispatcher - Starting workflow execution
[INFO] 2025-10-15 02:17:51,021 - Dispatcher - Executing node 'scene_detector' using executor 'opencv-scene-detect'
[INFO] 2025-10-15 02:17:51,821 - Dispatcher - Node 'scene_detector' completed successfully
[INFO] 2025-10-15 02:17:51,822 - Dispatcher - Executing node 'frame_extractor' using executor 'frame-extractor'
[INFO] 2025-10-15 02:17:51,823 - Dispatcher - Executing node 'speech_to_text' using executor 'whisper-v3'
[INFO] 2025-10-15 02:17:52,223 - Dispatcher - Node 'speech_to_text' completed successfully
[INFO] 2025-10-15 02:17:52,523 - Dispatcher - Node 'frame_extractor' completed successfully
[INFO] 2025-10-15 02:17:52,523 - Dispatcher - Executing node 'object_detector' using executor 'yolov8-object-detect'
[INFO] 2025-10-15 02:17:53,224 - Dispatcher - Node 'object_detector' completed successfully
[INFO] 2025-10-15 02:17:53,224 - Dispatcher - Executing node 'question_answer' using executor 'llama-3.2'
[INFO] 2025-10-15 02:17:54,124 - Dispatcher - Node 'question_answer' completed successfully
[INFO] 2025-10-15 02:17:54,124 - Dispatcher - Workflow execution completed

Final workflow outputs:
 - answer: simulated_result_from_question_answer_answer
```

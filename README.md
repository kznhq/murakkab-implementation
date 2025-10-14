# Murakkab Implementation
Implementing Murakkab from the [paper](https://arxiv.org/pdf/2508.18298)

## Architecture/Sequence of events
1. workflow (JSON)
2. define executor_library.Executor's and a executor_library.ExecutorLibrary() and use executor_library.register_executor(Executor) to register the Executors
    - doesn't have to be after building generic DAG, could be before step 2 or 1 too
3. define a workflow_orchestrator.WorkflowOrchestrator(ExecutorLibrary) then WorkflowOrchestrator.build_logical_plan(workflow)
    - this builds a DAG with nodes for each task in workflow and candidates for each node
4. define some profiles, resources, SLOs then use workflow_optimizer.WorkflowOptimizer(profiles: dict, resources: dict) and WorkflowOptimizer.optimize(logical DAG, SLOs)
    - examples in main.py but basically a bunch of dicts defining profiles of the executors, the resources, and the SLOs
    - optimize() will select one of the candidates to use for actual execution
    - TODO: currently just does greedy approach of using first candidate listed, need to flesh it out to actually optimize
5. TODO: actually execute the given DAG

## To run:
main.py goes through an example. Make changes there then run the main.py file.

## Other TODOs maybe:
- make REST endpoints or equivalent so that you can register workflows and execute workflows with different calls
- scaling?
- dynamic profiling that updates the profiles used by optimizer
    - also just measuring performance/profiling in general

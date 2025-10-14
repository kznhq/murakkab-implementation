from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass(frozen=True)
class Executor:
    """
    Represents a registered executor (model or tool) with metadata.

    Attributes:
        id (str): Unique identifier for the executor.
        type (str): Category of the executor (e.g., 'mlmodel', 'tool', 'llm').
        inputs (List[str]): List of supported input data types or modalities.
        outputs (List[str]): List of output data types produced.
        parameters (Optional[Dict[str, str]]): Optional dict of customizable parameters.
        resources (Optional[Dict[str, str]]): Optional dict describing resource requirements.
    """
    id: str
    type: str
    inputs: List[str]
    outputs: List[str]
    parameters: Optional[Dict[str, str]] = field(default_factory=dict)
    resources: Optional[Dict[str, str]] = field(default_factory=dict)


class ExecutorAlreadyExistsError(Exception):
    """Raised when attempting to register an executor with a duplicate ID."""
    pass


class ExecutorNotFoundError(Exception):
    """Raised when an executor with the specified ID is not found."""
    pass


class ExecutorLibrary:
    """
    Manages registration and lookup of executors in an AI workflow system.
    """

    def __init__(self):
        """Initialize an empty executor registry."""
        self._executors: Dict[str, Executor] = {}

    def register_executor(self, executor: Executor) -> None:
        """
        Register a new executor.

        Args:
            executor (Executor): Executor instance to register.

        Raises:
            ExecutorAlreadyExistsError: If an executor with the same ID already exists.
        """
        if executor.id in self._executors:
            raise ExecutorAlreadyExistsError(
                f"Executor with id '{executor.id}' already exists."
            )
        self._executors[executor.id] = executor

    def get_executor_by_id(self, executor_id: str) -> Executor:
        """
        Retrieve an executor by its unique ID.

        Args:
            executor_id (str): ID of the executor to retrieve.

        Returns:
            Executor: The corresponding executor instance.

        Raises:
            ExecutorNotFoundError: If no executor with the specified ID exists.
        """
        try:
            return self._executors[executor_id]
        except KeyError:
            raise ExecutorNotFoundError(f"Executor with id '{executor_id}' not found.")

    def query_executors(
        self,
        type: Optional[str] = None,
        input_type: Optional[str] = None,
        output_type: Optional[str] = None
    ) -> List[Executor]:
        """
        Query executors by type, input type, or output type.

        Args:
            type (Optional[str]): Filter by executor type.
            input_type (Optional[str]): Filter by supported input type.
            output_type (Optional[str]): Filter by supported output type.

        Returns:
            List[Executor]: List of executors matching all provided filters.
        """
        results = list(self._executors.values())

        if type:
            results = [ex for ex in results if ex.type == type]
        if input_type:
            results = [ex for ex in results if input_type in ex.inputs]
        if output_type:
            results = [ex for ex in results if output_type in ex.outputs]

        return results

    def list_all_executors(self) -> List[Executor]:
        """
        List all registered executors.

        Returns:
            List[Executor]: All executors in the registry.
        """
        return list(self._executors.values())


# Example usage
if __name__ == "__main__":
    exec_lib = ExecutorLibrary()

    # Register a new executor
    whisper_executor = Executor(
        id='whisper-v3',
        type='mlmodel',
        inputs=['audio'],
        outputs=['text'],
        parameters={'language': 'str', 'beam_size': 'int'},
        resources={'device': 'gpu', 'memory_gb': '12'}
    )
    exec_lib.register_executor(whisper_executor)

    # Retrieve by ID
    retrieved = exec_lib.get_executor_by_id('whisper-v3')
    print(f"Retrieved Executor: {retrieved}")

    # Query by input type
    audio_executors = exec_lib.query_executors(input_type='audio')
    print(f"Executors that accept audio: {[e.id for e in audio_executors]}")

    # List all executors
    all_executors = exec_lib.list_all_executors()
    print(f"All executors: {[e.id for e in all_executors]}")


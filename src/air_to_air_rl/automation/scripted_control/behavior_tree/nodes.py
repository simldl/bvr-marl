"""
Behavior tree nodes for tactical decision making.
Implements standard behavior tree patterns for air combat AI.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional


class NodeStatus(Enum):
    """Status returned by behavior tree nodes."""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Base class for all behavior tree nodes."""

    def __init__(self, name: str):
        self.name = name
        self.parent: BehaviorNode | None = None
        self.children: list[BehaviorNode] = []
        self.blackboard: dict[str, Any] = {}

    def add_child(self, child: "BehaviorNode"):
        """Add a child node."""
        child.parent = self
        child.blackboard = self.blackboard  # Share blackboard
        self.children.append(child)

    @abstractmethod
    def tick(self, context: dict[str, Any]) -> NodeStatus:
        """Execute the node's behavior."""
        pass

    def reset(self):
        """Reset the node state."""
        for child in self.children:
            child.reset()


class CompositeNode(BehaviorNode):
    """Base class for nodes with multiple children."""

    def __init__(self, name: str):
        super().__init__(name)
        self.current_child = 0

    def reset(self):
        super().reset()
        self.current_child = 0


class SelectorNode(CompositeNode):
    """Selector (OR) node - succeeds if any child succeeds."""

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        for i in range(self.current_child, len(self.children)):
            self.current_child = i
            status = self.children[i].tick(context)

            if status == NodeStatus.SUCCESS:
                self.current_child = 0
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            # Continue to next child on FAILURE

        # All children failed
        self.current_child = 0
        return NodeStatus.FAILURE


class SequenceNode(CompositeNode):
    """Sequence (AND) node - succeeds if all children succeed."""

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        for i in range(self.current_child, len(self.children)):
            self.current_child = i
            status = self.children[i].tick(context)

            if status == NodeStatus.FAILURE:
                self.current_child = 0
                return NodeStatus.FAILURE
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            # Continue to next child on SUCCESS

        # All children succeeded
        self.current_child = 0
        return NodeStatus.SUCCESS


class ParallelNode(CompositeNode):
    """Parallel node - executes all children simultaneously."""

    def __init__(self, name: str, success_threshold: int = 1, failure_threshold: int = 1):
        super().__init__(name)
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        success_count = 0
        failure_count = 0
        running_count = 0

        for child in self.children:
            status = child.tick(context)
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
            elif status == NodeStatus.RUNNING:
                running_count += 1

        if success_count >= self.success_threshold:
            return NodeStatus.SUCCESS
        elif failure_count >= self.failure_threshold:
            return NodeStatus.FAILURE
        else:
            return NodeStatus.RUNNING


class ConditionNode(BehaviorNode):
    """Condition node - checks a condition and returns success/failure."""

    def __init__(self, name: str, condition_func: Callable[[dict[str, Any]], bool]):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        try:
            if self.condition_func(context):
                return NodeStatus.SUCCESS
            else:
                return NodeStatus.FAILURE
        except Exception as e:
            self.blackboard[f"{self.name}_error"] = str(e)
            return NodeStatus.FAILURE


class ActionNode(BehaviorNode):
    """Action node - performs an action."""

    def __init__(self, name: str, action_func: Callable[[dict[str, Any]], NodeStatus]):
        super().__init__(name)
        self.action_func = action_func

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        try:
            return self.action_func(context)
        except Exception as e:
            self.blackboard[f"{self.name}_error"] = str(e)
            return NodeStatus.FAILURE


class DecoratorNode(BehaviorNode):
    """Base class for decorator nodes (single child)."""

    def __init__(self, name: str):
        super().__init__(name)

    def add_child(self, child: BehaviorNode):
        if len(self.children) >= 1:
            raise ValueError("Decorator nodes can only have one child")
        super().add_child(child)


class InverterNode(DecoratorNode):
    """Inverter node - inverts the result of its child."""

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        if not self.children:
            return NodeStatus.FAILURE

        status = self.children[0].tick(context)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.RUNNING


class RepeatNode(DecoratorNode):
    """Repeat node - repeats its child a specified number of times."""

    def __init__(self, name: str, repeat_count: int = -1):
        super().__init__(name)
        self.repeat_count = repeat_count  # -1 for infinite
        self.current_count = 0

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        if not self.children:
            return NodeStatus.FAILURE

        if self.repeat_count > 0 and self.current_count >= self.repeat_count:
            self.current_count = 0
            return NodeStatus.SUCCESS

        status = self.children[0].tick(context)
        if status == NodeStatus.SUCCESS or status == NodeStatus.FAILURE:
            self.current_count += 1
            if self.repeat_count > 0 and self.current_count >= self.repeat_count:
                self.current_count = 0
                return NodeStatus.SUCCESS
            else:
                self.children[0].reset()
                return NodeStatus.RUNNING

        return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.current_count = 0

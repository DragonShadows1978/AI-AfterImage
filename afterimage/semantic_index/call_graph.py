"""
CallGraph - Function/method call relationship tracking.

Provides:
- Building call graphs from parsed code
- Finding callers and callees
- Computing transitive call chains
- Detecting recursive calls
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .models import CallSite, LocationRange

if TYPE_CHECKING:
    from .performance import LazyCallGraphClosure


@dataclass
class CallGraphNode:
    """A node in the call graph representing a callable."""
    qualified_name: str
    display_name: str
    file_path: Optional[str] = None
    location: Optional[LocationRange] = None

    # Direct calls made by this function
    calls: Set[str] = field(default_factory=set)

    # Functions that call this function
    called_by: Set[str] = field(default_factory=set)

    # Call sites for detailed information
    call_sites: List[CallSite] = field(default_factory=list)

    # Metadata
    is_method: bool = False
    is_async: bool = False
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "qualified_name": self.qualified_name,
            "display_name": self.display_name,
            "file_path": self.file_path,
            "calls": list(self.calls),
            "called_by": list(self.called_by),
            "call_count": len(self.call_sites),
            "is_method": self.is_method,
            "class_name": self.class_name,
        }


class CallGraph:
    """
    Graph of function/method call relationships.

    Supports:
    - Building from parsed symbol tables
    - Finding callers and callees
    - Computing call chains
    - Detecting recursion
    """

    def __init__(self, use_lazy_closure: bool = False):
        self.nodes: Dict[str, CallGraphNode] = {}
        self._transitive_computed = False
        self._transitive_calls: Dict[str, Set[str]] = {}
        self._transitive_callers: Dict[str, Set[str]] = {}

        # Optional lazy closure for improved performance
        self._use_lazy_closure = use_lazy_closure
        self._lazy_closure: Optional['LazyCallGraphClosure'] = None
        if use_lazy_closure:
            from .performance import LazyCallGraphClosure
            self._lazy_closure = LazyCallGraphClosure()

    def add_function(
        self,
        qualified_name: str,
        display_name: str,
        file_path: Optional[str] = None,
        location: Optional[LocationRange] = None,
        is_method: bool = False,
        is_async: bool = False,
        class_name: Optional[str] = None,
    ) -> CallGraphNode:
        """Add a function/method to the call graph."""
        if qualified_name in self.nodes:
            return self.nodes[qualified_name]

        node = CallGraphNode(
            qualified_name=qualified_name,
            display_name=display_name,
            file_path=file_path,
            location=location,
            is_method=is_method,
            is_async=is_async,
            class_name=class_name,
        )
        self.nodes[qualified_name] = node
        self._invalidate_transitive()
        return node

    def add_call(
        self,
        caller: str,
        callee: str,
        call_site: Optional[CallSite] = None,
    ) -> None:
        """Record a call from one function to another."""
        # Ensure both nodes exist
        if caller not in self.nodes:
            self.add_function(caller, caller.split(".")[-1])
        if callee not in self.nodes:
            self.add_function(callee, callee.split(".")[-1])

        # Add the call relationship
        self.nodes[caller].calls.add(callee)
        self.nodes[callee].called_by.add(caller)

        # Store call site details
        if call_site:
            self.nodes[caller].call_sites.append(call_site)

        self._invalidate_transitive()

        # Sync with lazy closure if enabled
        if self._lazy_closure:
            self._lazy_closure.add_call(caller, callee)

    def _invalidate_transitive(self) -> None:
        """Invalidate cached transitive computations."""
        self._transitive_computed = False
        self._transitive_calls.clear()
        self._transitive_callers.clear()

        # Invalidate lazy closure if enabled
        if self._lazy_closure:
            self._lazy_closure.invalidate()

    def get_direct_calls(self, qualified_name: str) -> Set[str]:
        """Get functions directly called by a function."""
        if qualified_name not in self.nodes:
            return set()
        return self.nodes[qualified_name].calls.copy()

    def get_direct_callers(self, qualified_name: str) -> Set[str]:
        """Get functions that directly call a function."""
        if qualified_name not in self.nodes:
            return set()
        return self.nodes[qualified_name].called_by.copy()

    def compute_transitive(self) -> None:
        """Compute transitive closure of call relationships."""
        if self._transitive_computed:
            return

        # Compute transitive calls (what does each function eventually call)
        for name in self.nodes:
            visited = set()
            stack = list(self.nodes[name].calls)

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current in self.nodes:
                    stack.extend(self.nodes[current].calls - visited)

            self._transitive_calls[name] = visited

        # Compute transitive callers (what functions eventually call this)
        for name in self.nodes:
            visited = set()
            stack = list(self.nodes[name].called_by)

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current in self.nodes:
                    stack.extend(self.nodes[current].called_by - visited)

            self._transitive_callers[name] = visited

        self._transitive_computed = True

    def get_transitive_calls(self, qualified_name: str) -> Set[str]:
        """Get all functions transitively called by a function."""
        # Use lazy closure if enabled for on-demand computation
        if self._lazy_closure:
            return self._lazy_closure.get_transitive_calls(qualified_name)
        self.compute_transitive()
        return self._transitive_calls.get(qualified_name, set()).copy()

    def get_transitive_callers(self, qualified_name: str) -> Set[str]:
        """Get all functions that transitively call a function."""
        # Use lazy closure if enabled for on-demand computation
        if self._lazy_closure:
            return self._lazy_closure.get_transitive_callers(qualified_name)
        self.compute_transitive()
        return self._transitive_callers.get(qualified_name, set()).copy()

    def is_recursive(self, qualified_name: str) -> bool:
        """Check if a function is recursive (directly or indirectly)."""
        if qualified_name not in self.nodes:
            return False

        # Direct recursion
        if qualified_name in self.nodes[qualified_name].calls:
            return True

        # Indirect recursion
        self.compute_transitive()
        return qualified_name in self._transitive_calls.get(qualified_name, set())

    def find_recursive_functions(self) -> List[str]:
        """Find all recursive functions in the graph."""
        return [name for name in self.nodes if self.is_recursive(name)]

    def get_call_chain(
        self,
        from_func: str,
        to_func: str,
        max_depth: int = 10
    ) -> Optional[List[str]]:
        """
        Find a call chain from one function to another.

        Returns the shortest path, or None if no path exists.
        """
        if from_func not in self.nodes or to_func not in self.nodes:
            return None

        # BFS for shortest path
        from collections import deque
        visited = {from_func}
        queue = deque([(from_func, [from_func])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == to_func:
                return path

            if current in self.nodes:
                for callee in self.nodes[current].calls:
                    if callee not in visited:
                        visited.add(callee)
                        queue.append((callee, path + [callee]))

        return None

    def get_entry_points(self) -> List[str]:
        """Find functions that are never called (potential entry points)."""
        return [
            name for name, node in self.nodes.items()
            if not node.called_by
        ]

    def get_leaf_functions(self) -> List[str]:
        """Find functions that don't call anything else."""
        return [
            name for name, node in self.nodes.items()
            if not node.calls
        ]

    def get_call_sites(
        self,
        caller: str,
        callee: Optional[str] = None
    ) -> List[CallSite]:
        """Get call sites for a caller, optionally filtered to specific callee."""
        if caller not in self.nodes:
            return []

        sites = self.nodes[caller].call_sites
        if callee:
            return [s for s in sites if s.callee_qualified_name == callee]
        return sites.copy()

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "entry_points": self.get_entry_points(),
            "leaf_functions": self.get_leaf_functions(),
            "recursive_functions": self.find_recursive_functions(),
            "total_functions": len(self.nodes),
            "total_calls": sum(len(n.calls) for n in self.nodes.values()),
        }

    def to_dot(self, highlight_recursive: bool = True) -> str:
        """Generate DOT format for visualization."""
        lines = ["digraph calls {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")

        recursive = set(self.find_recursive_functions()) if highlight_recursive else set()

        # Add nodes
        for name, node in self.nodes.items():
            safe_name = name.replace(".", "_").replace("<", "_").replace(">", "_")
            label = node.display_name
            if node.class_name:
                label = f"{node.class_name}.{label}"

            style = ""
            if name in recursive:
                style = ', style=filled, fillcolor=yellow'

            lines.append(f'  {safe_name} [label="{label}"{style}];')

        # Add edges
        for name, node in self.nodes.items():
            safe_from = name.replace(".", "_").replace("<", "_").replace(">", "_")
            for callee in node.calls:
                safe_to = callee.replace(".", "_").replace("<", "_").replace(">", "_")

                # Highlight self-calls
                color = ""
                if name == callee:
                    color = " [color=red, style=bold]"

                lines.append(f"  {safe_from} -> {safe_to}{color};")

        lines.append("}")
        return "\n".join(lines)

    def merge(self, other: 'CallGraph') -> None:
        """Merge another call graph into this one."""
        for name, node in other.nodes.items():
            if name not in self.nodes:
                self.nodes[name] = CallGraphNode(
                    qualified_name=node.qualified_name,
                    display_name=node.display_name,
                    file_path=node.file_path,
                    location=node.location,
                    is_method=node.is_method,
                    is_async=node.is_async,
                    class_name=node.class_name,
                )

            # Merge calls
            self.nodes[name].calls.update(node.calls)
            self.nodes[name].call_sites.extend(node.call_sites)

        # Update called_by relationships
        for name, node in self.nodes.items():
            for callee in node.calls:
                if callee in self.nodes:
                    self.nodes[callee].called_by.add(name)

        self._invalidate_transitive()

    def __repr__(self) -> str:
        return f"CallGraph(nodes={len(self.nodes)}, calls={sum(len(n.calls) for n in self.nodes.values())})"

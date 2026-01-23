"""
Import Graph Analysis - Transitive dependency resolution and cycle detection.

Provides:
- ImportGraph: Build and analyze import/dependency graphs
- Transitive closure computation
- Circular import detection
- Path resolution for relative imports
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import ImportInfo


@dataclass
class ImportNode:
    """A node in the import dependency graph."""
    module: str
    file_path: Optional[str] = None
    imports: List[ImportInfo] = field(default_factory=list)

    # Computed during graph analysis
    direct_dependencies: Set[str] = field(default_factory=set)
    transitive_dependencies: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "file_path": self.file_path,
            "imports": [i.to_dict() for i in self.imports],
            "direct_dependencies": list(self.direct_dependencies),
            "transitive_dependencies": list(self.transitive_dependencies),
        }


@dataclass
class CycleInfo:
    """Information about a detected circular import."""
    cycle_path: List[str]  # Module names in the cycle
    involved_modules: Set[str]

    @property
    def cycle_string(self) -> str:
        """Human-readable cycle representation."""
        return " -> ".join(self.cycle_path + [self.cycle_path[0]])

    def to_dict(self) -> dict:
        return {
            "cycle_path": self.cycle_path,
            "cycle_string": self.cycle_string,
            "involved_modules": list(self.involved_modules),
        }


class ImportGraph:
    """
    Dependency graph for analyzing imports across multiple files.

    Supports:
    - Building graph from parsed AST results
    - Computing transitive closure of dependencies
    - Detecting circular imports
    - Resolving relative imports to absolute paths
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the import graph.

        Args:
            base_path: Base directory for resolving relative imports
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.nodes: Dict[str, ImportNode] = {}
        self._cycles: Optional[List[CycleInfo]] = None
        self._transitive_computed = False

    def add_module(
        self,
        module_name: str,
        imports: List[ImportInfo],
        file_path: Optional[str] = None
    ):
        """
        Add a module and its imports to the graph.

        Args:
            module_name: Name of the module (e.g., "mypackage.utils")
            imports: List of ImportInfo from parsing the module
            file_path: Optional path to the source file
        """
        node = ImportNode(
            module=module_name,
            file_path=file_path,
            imports=imports,
        )

        # Extract direct dependencies
        for imp in imports:
            resolved = self.resolve_import(imp.module, module_name)
            node.direct_dependencies.add(resolved)

        self.nodes[module_name] = node

        # Invalidate cached computations
        self._cycles = None
        self._transitive_computed = False

    def resolve_import(
        self,
        import_path: str,
        from_module: Optional[str] = None
    ) -> str:
        """
        Resolve a potentially relative import to an absolute module path.

        Args:
            import_path: The import string (may be relative like ".utils")
            from_module: The module containing the import

        Returns:
            Resolved absolute module path
        """
        if not import_path.startswith("."):
            # Already absolute
            return import_path

        if not from_module:
            # Can't resolve relative import without context
            return import_path

        # Count leading dots
        dots = 0
        for char in import_path:
            if char == ".":
                dots += 1
            else:
                break

        # Get base module path
        parts = from_module.split(".")

        # Go up 'dots' levels (. = same package, .. = parent, etc.)
        # For a module like "mypackage.sub.main":
        # . (1 dot) -> same package -> ["mypackage", "sub"]
        # .. (2 dots) -> parent package -> ["mypackage"]
        if dots >= len(parts):
            # Invalid relative import (goes above top-level)
            return import_path

        # Remove 'dots' parts from the end (the module itself + parents)
        base_parts = parts[:len(parts) - dots]

        # Append the rest of the import path
        rest = import_path[dots:]
        if rest:
            base_parts.append(rest)

        return ".".join(base_parts) if base_parts else import_path

    def compute_transitive_closure(self):
        """
        Compute transitive dependencies for all modules.

        Uses BFS from each node to find all reachable dependencies.
        """
        if self._transitive_computed:
            return

        for module_name, node in self.nodes.items():
            visited = set()
            queue = deque(node.direct_dependencies)

            while queue:
                dep = queue.popleft()
                if dep in visited:
                    continue
                visited.add(dep)

                # Add to transitive dependencies
                node.transitive_dependencies.add(dep)

                # If this dependency is in the graph, add its dependencies
                if dep in self.nodes:
                    for sub_dep in self.nodes[dep].direct_dependencies:
                        if sub_dep not in visited:
                            queue.append(sub_dep)

        self._transitive_computed = True

    def get_transitive_dependencies(self, module_name: str) -> Set[str]:
        """
        Get all transitive dependencies of a module.

        Args:
            module_name: The module to get dependencies for

        Returns:
            Set of all modules this module depends on (directly or transitively)
        """
        self.compute_transitive_closure()

        if module_name not in self.nodes:
            return set()

        return self.nodes[module_name].transitive_dependencies.copy()

    def get_direct_dependencies(self, module_name: str) -> Set[str]:
        """
        Get direct (immediate) dependencies of a module.

        Args:
            module_name: The module to get dependencies for

        Returns:
            Set of modules directly imported by this module
        """
        if module_name not in self.nodes:
            return set()

        return self.nodes[module_name].direct_dependencies.copy()

    def detect_cycles(self) -> List[CycleInfo]:
        """
        Detect all circular import chains in the graph.

        Uses Tarjan's algorithm to find strongly connected components,
        which represent circular dependency groups.

        Returns:
            List of CycleInfo objects describing each cycle
        """
        if self._cycles is not None:
            return self._cycles

        cycles = []

        # Build adjacency list for only known modules
        adj: Dict[str, List[str]] = defaultdict(list)
        for module_name, node in self.nodes.items():
            for dep in node.direct_dependencies:
                if dep in self.nodes:  # Only consider deps we know about
                    adj[module_name].append(dep)

        # Tarjan's algorithm state
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = {}
        sccs = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj[v]:
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])

            # If v is a root node, pop the SCC
            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:  # Only consider non-trivial SCCs (cycles)
                    sccs.append(scc)

        for v in self.nodes:
            if v not in index:
                strongconnect(v)

        # Convert SCCs to CycleInfo objects
        for scc in sccs:
            # Find a cycle path within the SCC
            cycle_path = self._find_cycle_path(scc, adj)
            if cycle_path:
                cycles.append(CycleInfo(
                    cycle_path=cycle_path,
                    involved_modules=set(scc),
                ))

        self._cycles = cycles
        return cycles

    def _find_cycle_path(
        self,
        scc: List[str],
        adj: Dict[str, List[str]]
    ) -> List[str]:
        """Find a specific cycle path within a strongly connected component."""
        scc_set = set(scc)
        start = scc[0]

        # DFS to find a path back to start
        visited = set()
        path = []

        def dfs(node):
            if node in visited:
                return node == start and len(path) > 1

            visited.add(node)
            path.append(node)

            for neighbor in adj[node]:
                if neighbor in scc_set:
                    if dfs(neighbor):
                        return True

            path.pop()
            visited.remove(node)
            return False

        if dfs(start):
            return path

        # Fallback: return SCC members in order
        return scc

    def has_cycles(self) -> bool:
        """Check if the graph contains any circular imports."""
        return len(self.detect_cycles()) > 0

    def get_dependency_chain(
        self,
        from_module: str,
        to_module: str
    ) -> Optional[List[str]]:
        """
        Find the dependency chain from one module to another.

        Args:
            from_module: Starting module
            to_module: Target module

        Returns:
            List of modules in the dependency path, or None if no path exists
        """
        if from_module not in self.nodes:
            return None

        # BFS to find shortest path
        visited = {from_module}
        queue = deque([(from_module, [from_module])])

        while queue:
            current, path = queue.popleft()

            if current == to_module:
                return path

            if current in self.nodes:
                for dep in self.nodes[current].direct_dependencies:
                    if dep not in visited:
                        visited.add(dep)
                        queue.append((dep, path + [dep]))

        return None

    def get_reverse_dependencies(self, module_name: str) -> Set[str]:
        """
        Get all modules that depend on the given module.

        Args:
            module_name: The module to find dependents of

        Returns:
            Set of modules that import this module (directly or transitively)
        """
        self.compute_transitive_closure()

        dependents = set()
        for name, node in self.nodes.items():
            if module_name in node.transitive_dependencies:
                dependents.add(name)

        return dependents

    def to_dict(self) -> dict:
        """Convert the entire graph to a dictionary representation."""
        self.compute_transitive_closure()
        cycles = self.detect_cycles()

        return {
            "base_path": str(self.base_path),
            "modules": {name: node.to_dict() for name, node in self.nodes.items()},
            "cycles": [c.to_dict() for c in cycles],
            "has_cycles": len(cycles) > 0,
            "total_modules": len(self.nodes),
        }

    def to_dot(self) -> str:
        """
        Generate DOT format for visualization.

        Returns:
            DOT graph string suitable for Graphviz
        """
        lines = ["digraph imports {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")

        # Add nodes
        for module_name in self.nodes:
            safe_name = module_name.replace(".", "_")
            lines.append(f'  {safe_name} [label="{module_name}"];')

        # Add edges
        for module_name, node in self.nodes.items():
            safe_from = module_name.replace(".", "_")
            for dep in node.direct_dependencies:
                if dep in self.nodes:
                    safe_to = dep.replace(".", "_")
                    lines.append(f"  {safe_from} -> {safe_to};")

        # Highlight cycles
        cycles = self.detect_cycles()
        if cycles:
            lines.append("  // Cycle edges (highlighted)")
            for cycle in cycles:
                for i in range(len(cycle.cycle_path)):
                    from_mod = cycle.cycle_path[i]
                    to_mod = cycle.cycle_path[(i + 1) % len(cycle.cycle_path)]
                    safe_from = from_mod.replace(".", "_")
                    safe_to = to_mod.replace(".", "_")
                    lines.append(f'  {safe_from} -> {safe_to} [color=red, penwidth=2];')

        lines.append("}")
        return "\n".join(lines)


def build_import_graph_from_results(
    results: Dict[str, 'ASTResult'],
    base_path: Optional[str] = None
) -> ImportGraph:
    """
    Build an import graph from multiple parsed AST results.

    Args:
        results: Dictionary mapping module names to their ASTResult
        base_path: Base directory for resolving relative imports

    Returns:
        ImportGraph with all modules and their dependencies
    """
    graph = ImportGraph(base_path)

    for module_name, result in results.items():
        graph.add_module(
            module_name,
            result.imports,
            result.file_path,
        )

    return graph

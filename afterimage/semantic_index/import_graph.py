"""
ImportGraph - Track import relationships and exported symbols across modules.

Provides:
- Module dependency tracking
- Export/import resolution
- Cross-file symbol availability
- Circular import detection
- Star import expansion
- Re-export tracking
- External package marking
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .models import Symbol, SymbolKind


@dataclass
class ModuleExports:
    """Tracks what symbols a module exports."""
    module_name: str
    file_path: str

    # Explicitly exported symbols (in __all__ or public)
    public_symbols: Dict[str, Symbol] = field(default_factory=dict)

    # All symbols defined at module level
    all_symbols: Dict[str, Symbol] = field(default_factory=dict)

    # Re-exported symbols (from submodules)
    re_exports: Dict[str, str] = field(default_factory=dict)  # name -> source module

    # __all__ list if defined
    all_list: Optional[List[str]] = None

    # Star imports: what modules this module does "from X import *" on
    star_imports: List[str] = field(default_factory=list)

    # Expanded star imports: actual symbol names from star imports
    star_import_symbols: Dict[str, str] = field(default_factory=dict)  # name -> source module

    # Is this an external package (not in project)?
    is_external: bool = False

    def get_export(self, name: str) -> Optional[Symbol]:
        """Get an exported symbol by name."""
        # Check __all__ if defined
        if self.all_list is not None:
            if name not in self.all_list:
                return None

        # Check public symbols
        if name in self.public_symbols:
            return self.public_symbols[name]

        # Check all symbols as fallback
        if name in self.all_symbols:
            return self.all_symbols[name]

        # Check re-exports
        if name in self.re_exports:
            # Return a placeholder - caller will need to resolve
            return None

        return None

    def get_all_exported_names(self) -> Set[str]:
        """Get all names that this module exports."""
        if self.all_list is not None:
            return set(self.all_list)

        # Combine public symbols and re-exports
        names = set(self.public_symbols.keys())
        names.update(self.re_exports.keys())
        names.update(self.star_import_symbols.keys())
        return names

    def is_exported(self, name: str) -> bool:
        """Check if a name is exported."""
        if self.all_list is not None:
            return name in self.all_list
        # Default: public names (no leading underscore)
        return not name.startswith('_')

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "file_path": self.file_path,
            "public_symbols": list(self.public_symbols.keys()),
            "all_symbols": list(self.all_symbols.keys()),
            "re_exports": self.re_exports,
            "star_imports": self.star_imports,
            "star_import_symbols": list(self.star_import_symbols.keys()),
            "has_all": self.all_list is not None,
            "is_external": self.is_external,
        }


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    module: str  # The module being imported from
    names: List[str]  # Names being imported (empty for 'import module')
    aliases: Dict[str, str] = field(default_factory=dict)  # name -> alias
    is_star_import: bool = False  # from module import *
    is_relative: bool = False  # from . import or from .. import
    relative_level: int = 0  # Number of dots in relative import

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "names": self.names,
            "aliases": self.aliases,
            "is_star_import": self.is_star_import,
            "is_relative": self.is_relative,
            "relative_level": self.relative_level,
        }


@dataclass
class ModuleImports:
    """Tracks what a module imports."""
    module_name: str
    file_path: str

    # List of import statements
    imports: List[ImportInfo] = field(default_factory=list)

    # Quick lookup: name -> (module, original_name)
    imported_names: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    def add_import(self, import_info: ImportInfo) -> None:
        """Add an import to the tracking."""
        self.imports.append(import_info)

        # Update name lookup
        if import_info.names:
            for name in import_info.names:
                alias = import_info.aliases.get(name, name)
                self.imported_names[alias] = (import_info.module, name)
        else:
            # import module or import module as alias
            alias = import_info.aliases.get(import_info.module, import_info.module)
            # For 'import foo.bar', the local name is 'foo'
            local_name = alias.split('.')[0]
            self.imported_names[local_name] = (import_info.module, '')

    def get_source_module(self, name: str) -> Optional[Tuple[str, str]]:
        """Get the source module and original name for an imported name."""
        return self.imported_names.get(name)

    def get_imported_modules(self) -> Set[str]:
        """Get all modules that this module imports from."""
        modules = set()
        for imp in self.imports:
            modules.add(imp.module)
        return modules

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "file_path": self.file_path,
            "imports": [i.to_dict() for i in self.imports],
            "imported_names": dict(self.imported_names),
        }


class ImportGraph:
    """
    Graph of import relationships between modules.

    Tracks:
    - What each module exports
    - What each module imports
    - Module dependency relationships
    - Cross-file symbol resolution
    """

    def __init__(self):
        # Module exports by module name
        self.exports: Dict[str, ModuleExports] = {}

        # Module imports by module name
        self.imports: Dict[str, ModuleImports] = {}

        # Module name -> file path mapping
        self.module_paths: Dict[str, str] = {}

        # File path -> module name mapping
        self.path_modules: Dict[str, str] = {}

        # Dependency graph: module -> modules it depends on
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Reverse dependency graph: module -> modules that depend on it
        self._dependents: Dict[str, Set[str]] = defaultdict(set)

    def register_module(
        self,
        module_name: str,
        file_path: str,
        symbols: List[Symbol],
        import_infos: List[ImportInfo],
        all_list: Optional[List[str]] = None,
    ) -> None:
        """
        Register a module with its exports and imports.

        Args:
            module_name: The module's fully qualified name
            file_path: Path to the module's file
            symbols: All symbols defined in the module
            import_infos: Import statements in the module
            all_list: Contents of __all__ if defined
        """
        # Remove old registration if exists
        if module_name in self.exports:
            self.unregister_module(module_name)

        # Update path mappings
        self.module_paths[module_name] = file_path
        self.path_modules[file_path] = module_name

        # Build exports
        exports = ModuleExports(
            module_name=module_name,
            file_path=file_path,
            all_list=all_list,
        )

        for symbol in symbols:
            # Skip non-module-level symbols
            if symbol.kind == SymbolKind.PARAMETER:
                continue

            exports.all_symbols[symbol.name] = symbol

            # Check if it's public
            if not symbol.name.startswith('_'):
                exports.public_symbols[symbol.name] = symbol

        self.exports[module_name] = exports

        # Build imports
        module_imports = ModuleImports(
            module_name=module_name,
            file_path=file_path,
        )

        for import_info in import_infos:
            module_imports.add_import(import_info)

            # Update dependency tracking
            imported_module = self._resolve_module_name(
                import_info.module,
                module_name,
                import_info.relative_level,
            )
            if imported_module:
                self._dependencies[module_name].add(imported_module)
                self._dependents[imported_module].add(module_name)

        self.imports[module_name] = module_imports

    def unregister_module(self, module_name: str) -> None:
        """Remove a module from the graph."""
        if module_name in self.exports:
            file_path = self.exports[module_name].file_path
            del self.exports[module_name]
            del self.path_modules[file_path]

        if module_name in self.imports:
            del self.imports[module_name]

        if module_name in self.module_paths:
            del self.module_paths[module_name]

        # Clean up dependency tracking
        if module_name in self._dependencies:
            for dep in self._dependencies[module_name]:
                self._dependents[dep].discard(module_name)
            del self._dependencies[module_name]

        if module_name in self._dependents:
            for dependent in self._dependents[module_name]:
                self._dependencies[dependent].discard(module_name)
            del self._dependents[module_name]

    def resolve_import(
        self,
        name: str,
        from_module: str,
    ) -> Optional[Symbol]:
        """
        Resolve an imported name to its definition.

        Args:
            name: The name as used in the importing module
            from_module: The module doing the importing

        Returns:
            The Symbol if found, None otherwise
        """
        if from_module not in self.imports:
            return None

        module_imports = self.imports[from_module]
        source_info = module_imports.get_source_module(name)

        if not source_info:
            return None

        source_module, original_name = source_info

        # Resolve the source module name
        resolved_module = self._resolve_module_name(source_module, from_module, 0)
        if not resolved_module or resolved_module not in self.exports:
            return None

        # Get the export
        exports = self.exports[resolved_module]
        target_name = original_name if original_name else name

        return exports.get_export(target_name)

    def get_exported_symbols(self, module_name: str) -> List[Symbol]:
        """Get all exported symbols from a module."""
        if module_name not in self.exports:
            return []

        exports = self.exports[module_name]
        if exports.all_list is not None:
            return [
                exports.all_symbols[name]
                for name in exports.all_list
                if name in exports.all_symbols
            ]

        return list(exports.public_symbols.values())

    def get_dependencies(self, module_name: str) -> Set[str]:
        """Get modules that a module depends on."""
        return self._dependencies.get(module_name, set()).copy()

    def get_dependents(self, module_name: str) -> Set[str]:
        """Get modules that depend on a module."""
        return self._dependents.get(module_name, set()).copy()

    def find_circular_imports(self) -> List[List[str]]:
        """Find all circular import chains."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._dependencies.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for module in self._dependencies:
            if module not in visited:
                dfs(module, [])

        return cycles

    def get_import_chain(
        self,
        from_module: str,
        to_module: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        """
        Find the import chain from one module to another.

        Returns the shortest path, or None if no path exists.
        """
        if from_module not in self._dependencies:
            return None

        from collections import deque
        visited = {from_module}
        queue = deque([(from_module, [from_module])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == to_module:
                return path

            for dep in self._dependencies.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, path + [dep]))

        return None

    def get_module_by_path(self, file_path: str) -> Optional[str]:
        """Get module name from file path."""
        return self.path_modules.get(file_path)

    def get_path_by_module(self, module_name: str) -> Optional[str]:
        """Get file path from module name."""
        return self.module_paths.get(module_name)

    def _resolve_module_name(
        self,
        module: str,
        from_module: str,
        relative_level: int,
    ) -> Optional[str]:
        """Resolve a potentially relative module name."""
        if relative_level == 0:
            return module

        # Relative import
        parts = from_module.split('.')
        if relative_level > len(parts):
            return None  # Invalid relative import

        base_parts = parts[:-relative_level] if relative_level > 0 else parts
        if module:
            return '.'.join(base_parts + [module])
        return '.'.join(base_parts)

    def resolve_complex_relative_import(
        self,
        import_path: str,
        from_module: str,
        from_file_path: str,
    ) -> Optional[str]:
        """
        Resolve complex relative imports like 'from ../../module import symbol'.

        Args:
            import_path: The import path (e.g., "..utils.helpers")
            from_module: The module doing the importing
            from_file_path: File path of the importing module

        Returns:
            Resolved absolute module name, or None if unresolvable
        """
        # Count leading dots
        relative_level = 0
        for char in import_path:
            if char == '.':
                relative_level += 1
            else:
                break

        # Get the module part after the dots
        module_part = import_path[relative_level:]

        # Use the standard resolution with the computed level
        return self._resolve_module_name(module_part, from_module, relative_level)

    def expand_star_import(
        self,
        source_module: str,
        target_module: str,
    ) -> Dict[str, Symbol]:
        """
        Expand a star import to get all symbols it brings in.

        Args:
            source_module: The module being star-imported from
            target_module: The module doing the star import

        Returns:
            Dict mapping symbol names to their Symbol objects
        """
        if source_module not in self.exports:
            return {}

        exports = self.exports[source_module]
        expanded = {}

        # Get all exported names
        exported_names = exports.get_all_exported_names()

        for name in exported_names:
            # Get the symbol
            symbol = exports.get_export(name)
            if symbol:
                expanded[name] = symbol

        # Track the star import in the target module's exports
        if target_module in self.exports:
            self.exports[target_module].star_imports.append(source_module)
            for name in expanded:
                self.exports[target_module].star_import_symbols[name] = source_module

        return expanded

    def register_re_export(
        self,
        module_name: str,
        symbol_name: str,
        source_module: str,
    ) -> None:
        """
        Register a re-export (when module A re-exports symbols from module B).

        Args:
            module_name: The module doing the re-export
            symbol_name: The name being re-exported
            source_module: The original source of the symbol
        """
        if module_name not in self.exports:
            return

        self.exports[module_name].re_exports[symbol_name] = source_module

    def resolve_re_export(
        self,
        module_name: str,
        symbol_name: str,
        visited: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, Symbol]]:
        """
        Resolve a re-exported symbol to its original definition.

        Args:
            module_name: The module where the re-export is accessed
            symbol_name: The symbol name to resolve
            visited: Set of already visited modules (for cycle detection)

        Returns:
            Tuple of (source_module, Symbol) or None if not found
        """
        if visited is None:
            visited = set()

        if module_name in visited:
            return None  # Cycle detected
        visited.add(module_name)

        if module_name not in self.exports:
            return None

        exports = self.exports[module_name]

        # First check direct symbols
        if symbol_name in exports.public_symbols:
            return (module_name, exports.public_symbols[symbol_name])

        if symbol_name in exports.all_symbols:
            return (module_name, exports.all_symbols[symbol_name])

        # Check re-exports
        if symbol_name in exports.re_exports:
            source_module = exports.re_exports[symbol_name]
            return self.resolve_re_export(source_module, symbol_name, visited)

        # Check star import symbols
        if symbol_name in exports.star_import_symbols:
            source_module = exports.star_import_symbols[symbol_name]
            return self.resolve_re_export(source_module, symbol_name, visited)

        return None

    def mark_external_package(self, module_name: str) -> None:
        """
        Mark a module as an external package (not part of the project).

        Args:
            module_name: The module name to mark as external
        """
        if module_name not in self.exports:
            # Create a minimal exports entry for external packages
            self.exports[module_name] = ModuleExports(
                module_name=module_name,
                file_path="<external>",
                is_external=True,
            )
        else:
            self.exports[module_name].is_external = True

    def is_external_package(self, module_name: str) -> bool:
        """Check if a module is an external package."""
        # Known external packages
        known_external = {
            "os", "sys", "re", "json", "typing", "collections",
            "dataclasses", "pathlib", "functools", "itertools",
            "datetime", "math", "random", "time", "hashlib",
            "copy", "io", "logging", "unittest", "pytest",
            "numpy", "pandas", "requests", "flask", "django",
        }

        # Check if it's a known external or marked external
        root_module = module_name.split('.')[0]
        if root_module in known_external:
            return True

        if module_name in self.exports:
            return self.exports[module_name].is_external

        # If module isn't registered and isn't in our paths, assume external
        return module_name not in self.module_paths

    def resolve_import_with_fallback(
        self,
        name: str,
        from_module: str,
    ) -> Tuple[Optional[Symbol], str]:
        """
        Resolve an imported name with external package fallback.

        Args:
            name: The name as used in the importing module
            from_module: The module doing the importing

        Returns:
            Tuple of (Symbol or None, status) where status is:
            - "resolved": Successfully resolved to a symbol
            - "external": Symbol is from an external package
            - "unresolved": Could not resolve
        """
        # Try normal resolution first
        symbol = self.resolve_import(name, from_module)
        if symbol:
            return (symbol, "resolved")

        # Check if it might be from an external package
        if from_module not in self.imports:
            return (None, "unresolved")

        module_imports = self.imports[from_module]
        source_info = module_imports.get_source_module(name)

        if source_info:
            source_module, _ = source_info
            if self.is_external_package(source_module):
                self.mark_external_package(source_module)
                return (None, "external")

        return (None, "unresolved")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "modules": list(self.module_paths.keys()),
            "exports": {k: v.to_dict() for k, v in self.exports.items()},
            "imports": {k: v.to_dict() for k, v in self.imports.items()},
            "dependencies": {k: list(v) for k, v in self._dependencies.items()},
            "circular_imports": self.find_circular_imports(),
        }

    def __repr__(self) -> str:
        return f"ImportGraph(modules={len(self.module_paths)}, deps={sum(len(v) for v in self._dependencies.values())})"

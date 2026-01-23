"""
Performance Benchmarking Suite for AST Parsers.

Provides comprehensive benchmarking capabilities:
- Cold vs warm (incremental) parse time measurements
- Memory usage tracking for different file sizes
- Cross-language performance comparison reports
- Baseline metrics for regression testing
"""

import gc
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .parser import parse, get_supported_languages, ASTParserFactory


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    language: str
    source_size_bytes: int
    source_lines: int

    # Timing metrics (in milliseconds)
    cold_parse_ms: float
    warm_parse_ms: float  # Second parse (cached state)
    incremental_parse_ms: float  # After edit

    # Memory metrics (in bytes)
    peak_memory_bytes: int
    memory_delta_bytes: int  # Memory increase during parse

    # Parse results
    total_nodes: int
    function_count: int
    class_count: int
    import_count: int
    parse_confidence: float

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "source_size_bytes": self.source_size_bytes,
            "source_lines": self.source_lines,
            "cold_parse_ms": round(self.cold_parse_ms, 3),
            "warm_parse_ms": round(self.warm_parse_ms, 3),
            "incremental_parse_ms": round(self.incremental_parse_ms, 3),
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_delta_bytes": self.memory_delta_bytes,
            "total_nodes": self.total_nodes,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "parse_confidence": self.parse_confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report across multiple runs."""
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def generate_summary(self):
        """Generate summary statistics from results."""
        by_language: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_language.setdefault(r.language, []).append(r)

        self.summary = {
            "generated_at": datetime.now().isoformat(),
            "total_benchmarks": len(self.results),
            "languages": {},
        }

        for lang, lang_results in by_language.items():
            cold_times = [r.cold_parse_ms for r in lang_results]
            warm_times = [r.warm_parse_ms for r in lang_results]
            incr_times = [r.incremental_parse_ms for r in lang_results]
            memory_peaks = [r.peak_memory_bytes for r in lang_results]

            self.summary["languages"][lang] = {
                "benchmark_count": len(lang_results),
                "cold_parse_ms": {
                    "mean": round(statistics.mean(cold_times), 3),
                    "median": round(statistics.median(cold_times), 3),
                    "min": round(min(cold_times), 3),
                    "max": round(max(cold_times), 3),
                    "stddev": round(statistics.stdev(cold_times), 3) if len(cold_times) > 1 else 0,
                },
                "warm_parse_ms": {
                    "mean": round(statistics.mean(warm_times), 3),
                    "median": round(statistics.median(warm_times), 3),
                    "min": round(min(warm_times), 3),
                    "max": round(max(warm_times), 3),
                },
                "incremental_parse_ms": {
                    "mean": round(statistics.mean(incr_times), 3),
                    "median": round(statistics.median(incr_times), 3),
                    "min": round(min(incr_times), 3),
                    "max": round(max(incr_times), 3),
                },
                "peak_memory_bytes": {
                    "mean": int(statistics.mean(memory_peaks)),
                    "max": max(memory_peaks),
                },
            }

        return self.summary

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps({
            "summary": self.summary or self.generate_summary(),
            "results": [r.to_dict() for r in self.results],
        }, indent=2)

    def save(self, path: str):
        """Save report to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    def print_summary(self):
        """Print human-readable summary to stdout."""
        if not self.summary:
            self.generate_summary()

        print("\n" + "=" * 70)
        print("AST PARSER BENCHMARK REPORT")
        print("=" * 70)
        print(f"Generated: {self.summary['generated_at']}")
        print(f"Total benchmarks: {self.summary['total_benchmarks']}")
        print()

        for lang, stats in self.summary.get("languages", {}).items():
            print(f"\n--- {lang.upper()} ---")
            print(f"  Benchmarks: {stats['benchmark_count']}")
            print(f"  Cold Parse:        mean={stats['cold_parse_ms']['mean']:.2f}ms, "
                  f"median={stats['cold_parse_ms']['median']:.2f}ms")
            print(f"  Warm Parse:        mean={stats['warm_parse_ms']['mean']:.2f}ms, "
                  f"median={stats['warm_parse_ms']['median']:.2f}ms")
            print(f"  Incremental Parse: mean={stats['incremental_parse_ms']['mean']:.2f}ms, "
                  f"median={stats['incremental_parse_ms']['median']:.2f}ms")
            print(f"  Peak Memory:       mean={stats['peak_memory_bytes']['mean'] / 1024:.1f}KB, "
                  f"max={stats['peak_memory_bytes']['max'] / 1024:.1f}KB")

        print("\n" + "=" * 70)


class Benchmark:
    """Benchmark runner for AST parsers."""

    CODE_GENERATORS = {}

    def __init__(self):
        self.report = BenchmarkReport()
        self._setup_code_generators()

    def _setup_code_generators(self):
        """Set up code generators for each language."""
        self.CODE_GENERATORS = {
            "python": self._generate_python_code,
            "javascript": self._generate_javascript_code,
            "typescript": self._generate_typescript_code,
            "rust": self._generate_rust_code,
            "go": self._generate_go_code,
            "c": self._generate_c_code,
            "cpp": self._generate_cpp_code,
        }

    def _generate_python_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate Python code with specified complexity."""
        lines = ['"""Generated Python module for benchmarking."""', '', 'import os', 'from typing import List, Optional, Dict', '']

        for i in range(num_classes):
            lines.append(f'class Class{i}:')
            lines.append(f'    """Class {i} documentation."""')
            lines.append('')
            lines.append(f'    def __init__(self, value: int = {i}):')
            lines.append(f'        self.value = value')
            lines.append('')
            for j in range(3):
                lines.append(f'    def method_{j}(self, x: int) -> int:')
                lines.append(f'        """Method {j} doc."""')
                lines.append(f'        return self.value + x + {j}')
                lines.append('')

        for i in range(num_functions):
            lines.append(f'def function_{i}(a: int, b: str = "default") -> Optional[int]:')
            lines.append(f'    """Function {i} documentation."""')
            lines.append(f'    return a + {i}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_javascript_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate JavaScript code with specified complexity."""
        lines = ['// Generated JavaScript module for benchmarking', '',
                 "import { something } from 'some-module';", '']

        for i in range(num_classes):
            lines.append(f'/**')
            lines.append(f' * Class{i} documentation')
            lines.append(f' */')
            lines.append(f'class Class{i} {{')
            lines.append(f'    constructor(value = {i}) {{')
            lines.append(f'        this.value = value;')
            lines.append(f'    }}')
            lines.append('')
            for j in range(3):
                lines.append(f'    method_{j}(x) {{')
                lines.append(f'        return this.value + x + {j};')
                lines.append(f'    }}')
                lines.append('')
            lines.append('}')
            lines.append('')

        for i in range(num_functions):
            lines.append(f'/**')
            lines.append(f' * Function {i} documentation')
            lines.append(f' */')
            lines.append(f'function function_{i}(a, b = "default") {{')
            lines.append(f'    return a + {i};')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_typescript_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate TypeScript code with specified complexity."""
        lines = ['// Generated TypeScript module for benchmarking', '',
                 "import { Something } from 'some-module';", '',
                 'interface DataItem {', '    id: number;', '    name: string;', '}', '']

        for i in range(num_classes):
            lines.append(f'class Class{i} {{')
            lines.append(f'    private value: number;')
            lines.append('')
            lines.append(f'    constructor(value: number = {i}) {{')
            lines.append(f'        this.value = value;')
            lines.append(f'    }}')
            lines.append('')
            for j in range(3):
                lines.append(f'    public method_{j}(x: number): number {{')
                lines.append(f'        return this.value + x + {j};')
                lines.append(f'    }}')
                lines.append('')
            lines.append('}')
            lines.append('')

        for i in range(num_functions):
            lines.append(f'function function_{i}(a: number, b: string = "default"): number {{')
            lines.append(f'    return a + {i};')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_rust_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate Rust code with specified complexity."""
        lines = ['//! Generated Rust module for benchmarking', '', 'use std::collections::HashMap;', '']

        for i in range(num_classes):
            lines.append(f'/// Struct{i} documentation')
            lines.append(f'pub struct Struct{i} {{')
            lines.append(f'    pub value: i32,')
            lines.append('}')
            lines.append('')
            lines.append(f'impl Struct{i} {{')
            lines.append(f'    /// Create new instance')
            lines.append(f'    pub fn new(value: i32) -> Self {{')
            lines.append(f'        Self {{ value }}')
            lines.append(f'    }}')
            lines.append('')
            for j in range(3):
                lines.append(f'    pub fn method_{j}(&self, x: i32) -> i32 {{')
                lines.append(f'        self.value + x + {j}')
                lines.append(f'    }}')
                lines.append('')
            lines.append('}')
            lines.append('')

        for i in range(num_functions):
            lines.append(f'/// Function {i} documentation')
            lines.append(f'pub fn function_{i}(a: i32, b: &str) -> i32 {{')
            lines.append(f'    a + {i}')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_go_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate Go code with specified complexity."""
        lines = ['// Package benchmark contains generated code for benchmarking', 'package benchmark', '',
                 'import (', '    "fmt"', '    "strings"', ')', '']

        for i in range(num_classes):
            lines.append(f'// Struct{i} is a generated struct')
            lines.append(f'type Struct{i} struct {{')
            lines.append(f'    Value int')
            lines.append('}')
            lines.append('')
            lines.append(f'// NewStruct{i} creates a new instance')
            lines.append(f'func NewStruct{i}(value int) *Struct{i} {{')
            lines.append(f'    return &Struct{i}{{Value: value}}')
            lines.append('}')
            lines.append('')
            for j in range(3):
                lines.append(f'// Method{j} performs computation')
                lines.append(f'func (s *Struct{i}) Method{j}(x int) int {{')
                lines.append(f'    return s.Value + x + {j}')
                lines.append('}')
                lines.append('')

        for i in range(num_functions):
            lines.append(f'// Function{i} is a generated function')
            lines.append(f'func Function{i}(a int, b string) int {{')
            lines.append(f'    _ = b  // Unused parameter')
            lines.append(f'    return a + {i}')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_c_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate C code with specified complexity."""
        lines = ['/* Generated C code for benchmarking */', '',
                 '#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>', '']

        for i in range(num_classes):
            lines.append(f'/* Struct{i} documentation */')
            lines.append(f'typedef struct Struct{i} {{')
            lines.append(f'    int value;')
            lines.append(f'}} Struct{i};')
            lines.append('')
            lines.append(f'/* Create new Struct{i} */')
            lines.append(f'Struct{i}* new_struct_{i}(int value) {{')
            lines.append(f'    Struct{i}* s = malloc(sizeof(Struct{i}));')
            lines.append(f'    s->value = value;')
            lines.append(f'    return s;')
            lines.append('}')
            lines.append('')
            for j in range(3):
                lines.append(f'int struct_{i}_method_{j}(Struct{i}* s, int x) {{')
                lines.append(f'    return s->value + x + {j};')
                lines.append('}')
                lines.append('')

        for i in range(num_functions):
            lines.append(f'/* Function {i} documentation */')
            lines.append(f'int function_{i}(int a, const char* b) {{')
            lines.append(f'    (void)b;  /* Unused parameter */')
            lines.append(f'    return a + {i};')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _generate_cpp_code(self, num_functions: int = 50, num_classes: int = 10) -> str:
        """Generate C++ code with specified complexity."""
        lines = ['// Generated C++ code for benchmarking', '',
                 '#include <iostream>', '#include <string>', '#include <vector>', '#include <memory>', '']

        for i in range(num_classes):
            lines.append(f'/**')
            lines.append(f' * Class{i} documentation')
            lines.append(f' */')
            lines.append(f'class Class{i} {{')
            lines.append('private:')
            lines.append(f'    int value_;')
            lines.append('')
            lines.append('public:')
            lines.append(f'    explicit Class{i}(int value = {i}) : value_(value) {{}}')
            lines.append('')
            for j in range(3):
                lines.append(f'    int method_{j}(int x) const {{')
                lines.append(f'        return value_ + x + {j};')
                lines.append(f'    }}')
                lines.append('')
            lines.append('};')
            lines.append('')

        for i in range(num_functions):
            lines.append(f'// Function {i} documentation')
            lines.append(f'int function_{i}(int a, const std::string& b = "default") {{')
            lines.append(f'    (void)b;  // Unused parameter')
            lines.append(f'    return a + {i};')
            lines.append('}')
            lines.append('')

        return '\n'.join(lines)

    def _time_parse(self, source: str, language: str, file_path: Optional[str] = None) -> Tuple[float, Any]:
        """Time a single parse operation. Returns (time_ms, result)."""
        start = time.perf_counter()
        result = parse(source, language, file_path=file_path)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, result

    def _measure_memory(self, func: Callable) -> Tuple[Any, int, int]:
        """Measure memory usage of a function. Returns (result, peak_bytes, delta_bytes)."""
        gc.collect()
        tracemalloc.start()

        result = func()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return result, peak, current

    def benchmark_language(
        self,
        language: str,
        num_functions: int = 50,
        num_classes: int = 10,
        runs: int = 3
    ) -> BenchmarkResult:
        """
        Run benchmarks for a single language.

        Args:
            language: Language name
            num_functions: Number of functions to generate
            num_classes: Number of classes/structs to generate
            runs: Number of runs to average

        Returns:
            BenchmarkResult with aggregated metrics
        """
        if language not in self.CODE_GENERATORS:
            raise ValueError(f"No code generator for language: {language}")

        if language not in get_supported_languages():
            raise ValueError(f"Language not supported by parser: {language}")

        # Generate code
        generator = self.CODE_GENERATORS[language]
        source = generator(num_functions, num_classes)
        source_bytes = len(source.encode('utf-8'))
        source_lines = len(source.split('\n'))

        # Clear parser cache
        ASTParserFactory.clear_cache(language)

        file_path = f"benchmark_{language}_test.tmp"

        cold_times = []
        warm_times = []
        incr_times = []
        peak_memories = []
        memory_deltas = []

        for run in range(runs):
            # Clear cache before each cold parse
            ASTParserFactory.clear_cache(language)

            # Cold parse with memory measurement
            def do_cold_parse():
                return self._time_parse(source, language, file_path)

            (cold_time, result), peak_mem, mem_delta = self._measure_memory(do_cold_parse)
            cold_times.append(cold_time)
            peak_memories.append(peak_mem)
            memory_deltas.append(mem_delta)

            # Warm parse (same content, cached state)
            warm_time, _ = self._time_parse(source, language, file_path)
            warm_times.append(warm_time)

            # Incremental parse (modified content)
            modified_source = source + f"\n# Additional comment for run {run}\n"
            incr_time, _ = self._time_parse(modified_source, language, file_path)
            incr_times.append(incr_time)

        # Get final result for metadata
        final_result = parse(source, language)

        benchmark_result = BenchmarkResult(
            language=language,
            source_size_bytes=source_bytes,
            source_lines=source_lines,
            cold_parse_ms=statistics.mean(cold_times),
            warm_parse_ms=statistics.mean(warm_times),
            incremental_parse_ms=statistics.mean(incr_times),
            peak_memory_bytes=max(peak_memories),
            memory_delta_bytes=int(statistics.mean(memory_deltas)),
            total_nodes=final_result.total_nodes,
            function_count=len(final_result.functions),
            class_count=len(final_result.classes),
            import_count=len(final_result.imports),
            parse_confidence=final_result.parse_confidence,
        )

        self.report.add_result(benchmark_result)
        return benchmark_result

    def benchmark_all_languages(
        self,
        num_functions: int = 50,
        num_classes: int = 10,
        runs: int = 3
    ) -> BenchmarkReport:
        """
        Run benchmarks for all supported languages.

        Returns:
            BenchmarkReport with all results
        """
        for language in self.CODE_GENERATORS.keys():
            if language in get_supported_languages():
                try:
                    self.benchmark_language(language, num_functions, num_classes, runs)
                except Exception as e:
                    print(f"Warning: Benchmark failed for {language}: {e}")

        self.report.generate_summary()
        return self.report

    def benchmark_file_sizes(
        self,
        language: str = "python",
        sizes: List[int] = None,
        runs: int = 3
    ) -> BenchmarkReport:
        """
        Benchmark parsing with different file sizes.

        Args:
            language: Language to benchmark
            sizes: List of (functions, classes) tuples
            runs: Number of runs per size

        Returns:
            BenchmarkReport with size-based results
        """
        if sizes is None:
            sizes = [(10, 2), (50, 10), (100, 20), (200, 40)]

        for num_funcs, num_classes in sizes:
            self.benchmark_language(language, num_funcs, num_classes, runs)

        self.report.generate_summary()
        return self.report


def run_benchmark(
    languages: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    num_functions: int = 50,
    num_classes: int = 10,
    runs: int = 3
) -> BenchmarkReport:
    """
    Run benchmarks and optionally save results.

    Args:
        languages: List of languages to benchmark (None = all supported)
        output_path: Optional path to save JSON report
        num_functions: Number of functions per test
        num_classes: Number of classes per test
        runs: Number of runs per language

    Returns:
        BenchmarkReport with results
    """
    bench = Benchmark()

    if languages is None:
        report = bench.benchmark_all_languages(num_functions, num_classes, runs)
    else:
        for lang in languages:
            try:
                bench.benchmark_language(lang, num_functions, num_classes, runs)
            except Exception as e:
                print(f"Warning: Benchmark failed for {lang}: {e}")
        report = bench.report
        report.generate_summary()

    if output_path:
        report.save(output_path)

    return report


def check_performance_regression(
    baseline_path: str,
    current_report: BenchmarkReport,
    threshold_pct: float = 20.0
) -> List[str]:
    """
    Check for performance regressions against a baseline.

    Args:
        baseline_path: Path to baseline JSON report
        current_report: Current benchmark report
        threshold_pct: Percentage increase that counts as regression

    Returns:
        List of regression warnings
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    warnings = []
    baseline_summary = baseline.get("summary", {}).get("languages", {})
    current_summary = current_report.summary.get("languages", {})

    for lang, current_stats in current_summary.items():
        if lang not in baseline_summary:
            continue

        baseline_stats = baseline_summary[lang]

        # Check cold parse time
        baseline_cold = baseline_stats["cold_parse_ms"]["mean"]
        current_cold = current_stats["cold_parse_ms"]["mean"]
        if current_cold > baseline_cold * (1 + threshold_pct / 100):
            pct_increase = ((current_cold / baseline_cold) - 1) * 100
            warnings.append(
                f"{lang}: Cold parse regression {pct_increase:.1f}% "
                f"({baseline_cold:.2f}ms -> {current_cold:.2f}ms)"
            )

        # Check incremental parse time
        baseline_incr = baseline_stats["incremental_parse_ms"]["mean"]
        current_incr = current_stats["incremental_parse_ms"]["mean"]
        if current_incr > baseline_incr * (1 + threshold_pct / 100):
            pct_increase = ((current_incr / baseline_incr) - 1) * 100
            warnings.append(
                f"{lang}: Incremental parse regression {pct_increase:.1f}% "
                f"({baseline_incr:.2f}ms -> {current_incr:.2f}ms)"
            )

    return warnings


if __name__ == "__main__":
    print("AST Parser Benchmark Suite")
    print("=" * 70)

    # Run benchmarks for currently supported languages
    report = run_benchmark(
        languages=["python", "javascript", "rust"],
        num_functions=30,
        num_classes=5,
        runs=3
    )

    report.print_summary()

    # Save report
    output_path = Path(__file__).parent.parent / "benchmark_results.json"
    report.save(str(output_path))
    print(f"\nReport saved to: {output_path}")

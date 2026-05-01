#!/usr/bin/env python3
"""
Quick-start script: Run all compression benchmarks and analyze results.
"""

import sys
from pathlib import Path

# Add parent directory to path to import example modules
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_compression import CompressionBenchmark
from analyze_results import BenchmarkAnalyzer


def main():
    """Run full benchmark pipeline."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / "datasets" / "quran-uthmani.txt"
    results_dir = project_root / "compression_results"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("COMPRESSION BENCHMARK PIPELINE")
    print("=" * 80)
    
    # Run benchmarks
    print("\n[1/3] Running benchmarks...")
    benchmark = CompressionBenchmark(data_path=dataset_path, output_dir=results_dir)
    benchmark.run_all()
    
    # Save results
    print("\n[2/3] Saving results...")
    json_path = benchmark.save_results("json")
    csv_path = benchmark.save_results("csv")
    benchmark.print_summary()
    
    # Analyze results
    print("\n[3/3] Analyzing results...")
    analyzer = BenchmarkAnalyzer(str(json_path))
    analyzer.print_comparison_table()
    analyzer.print_ranking("compression_ratio", top_n=5)
    analyzer.print_ranking("compress_time_ms", top_n=5, ascending=True)
    analyzer.print_recommendations()
    
    # Generate HTML report
    html_path = analyzer.generate_html_report()
    
    print("\n" + "=" * 80)
    print("RESULTS SAVED:")
    print(f"  JSON:  {json_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  HTML:  {html_path}")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Analyze and visualize compression benchmark results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
import sys


class BenchmarkAnalyzer:
    """Analyze compression benchmark results."""
    
    def __init__(self, results_path: str):
        """
        Initialize analyzer.
        
        Args:
            results_path: Path to benchmark_results.json
        """
        self.results_path = Path(results_path)
        
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
    
    def get_algorithms(self) -> List[Dict]:
        """Get list of algorithm results."""
        return [a for a in self.data.get("algorithms", []) if "error" not in a or not a["error"]]
    
    def rank_by(self, metric: str, ascending: bool = False) -> List[Dict]:
        """
        Rank algorithms by metric.
        
        Args:
            metric: Metric name (e.g., "compression_ratio", "compress_time_ms")
            ascending: If True, rank ascending; else descending
            
        Returns:
            Sorted list of algorithm results
        """
        algos = self.get_algorithms()
        algos.sort(key=lambda x: x.get(metric, float('inf')), reverse=not ascending)
        return algos
    
    def print_ranking(self, metric: str, top_n: int = None, ascending: bool = False):
        """
        Print ranking for a metric.
        
        Args:
            metric: Metric name
            top_n: Show top N results (None for all)
            ascending: If True, rank ascending; else descending
        """
        ranked = self.rank_by(metric, ascending=ascending)
        if top_n:
            ranked = ranked[:top_n]
        
        print(f"\n{'Ranking by ' + metric.replace('_', ' ')}"
              f" (Top {top_n if top_n else 'all'})")
        print("=" * 60)
        
        for i, result in enumerate(ranked, 1):
            value = result.get(metric, "N/A")
            if isinstance(value, float):
                if "ratio" in metric or "percent" in metric:
                    print(f"{i:2}. {result['name']:<25} {value:>10.2f}")
                elif "time" in metric or "throughput" in metric:
                    print(f"{i:2}. {result['name']:<25} {value:>10.2f}")
                else:
                    print(f"{i:2}. {result['name']:<25} {value:>10.2f}")
            else:
                print(f"{i:2}. {result['name']:<25} {str(value):>10}")
    
    def print_comparison_table(self):
        """Print comprehensive comparison table."""
        algos = self.get_algorithms()
        
        print("\n" + "=" * 145)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("=" * 145)
        
        header = (f"{'Algorithm':<30} {'Ratio':>12} {'Comp.Time':>15} {'Decomp.Time':>15} "
                 f"{'Total Time':>15} {'Comp.MBps':>15} {'Space Saved':>15}")
        print(header)
        print("-" * 145)
        
        for result in sorted(algos, key=lambda x: x.get("compression_ratio", 0), reverse=True):
            ratio_str = f"{result.get('compression_ratio', 0):.2f}x"
            comp_time_str = f"{result.get('compress_time_ms', 0):.2f}ms"
            decomp_time_str = f"{result.get('decompress_time_ms', 0):.2f}ms"
            total_time_str = f"{result.get('total_time_ms', 0):.2f}ms"
            throughput_str = f"{result.get('throughput_compress_mbps', 0):.2f}"
            space_saved_str = f"{result.get('space_saved_percent', 0):.1f}%"
            
            print(
                f"{result['name']:<30} "
                f"{ratio_str:>12} "
                f"{comp_time_str:>15} "
                f"{decomp_time_str:>15} "
                f"{total_time_str:>15} "
                f"{throughput_str:>15} "
                f"{space_saved_str:>15}"
            )
        
        print("=" * 145)
    
    def get_best_algorithm(self, metric: str) -> Dict:
        """Get best algorithm for a metric."""
        ranked = self.rank_by(metric, ascending=("time" in metric and "throughput" not in metric))
        return ranked[0] if ranked else None
    
    def print_recommendations(self):
        """Print recommendations based on results."""
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        best_ratio = self.get_best_algorithm("compression_ratio")
        best_speed = self.get_best_algorithm("compress_time_ms")
        best_balanced = None
        
        # Find best balanced (ratio / compress_time)
        algos = self.get_algorithms()
        algos_with_scores = [
            (a, a.get("compression_ratio", 0) / max(a.get("compress_time_ms", 1), 0.001))
            for a in algos
        ]
        if algos_with_scores:
            best_balanced = max(algos_with_scores, key=lambda x: x[1])[0]
        
        print(f"Best compression ratio  : {best_ratio['name']} ({best_ratio.get('compression_ratio', 0):.2f}x)")
        print(f"Fastest compression    : {best_speed['name']} ({best_speed.get('compress_time_ms', 0):.2f}ms)")
        if best_balanced:
            print(f"Best balanced (ratio/speed): {best_balanced['name']}")
        
        print("=" * 60)
    
    def generate_html_report(self, output_path: str = None) -> str:
        """
        Generate HTML report.
        
        Args:
            output_path: Path to save HTML (default: results_dir/report.html)
            
        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = self.results_path.parent / "report.html"
        
        algos = self.get_algorithms()
        original_size = self.data.get("original_size", 0)
        
        # Sort by compression ratio
        algos.sort(key=lambda x: x.get("compression_ratio", 0), reverse=True)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Compression Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f0f0;
        }}
        .metric-info {{
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .best {{
            background-color: #fff9c4;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Compression Algorithm Benchmark Report</h1>
    
    <div class="metric-info">
        <p><strong>Dataset:</strong> {self.data.get('file', 'Unknown')}</p>
        <p><strong>Original Size:</strong> {original_size:,} bytes ({original_size/(1024**2):.2f} MB)</p>
    </div>
    
    <h2>Algorithm Comparison</h2>
    <table>
        <tr>
            <th>Algorithm</th>
            <th>Compressed Size (B)</th>
            <th>Compression Ratio</th>
            <th>Space Saved (%)</th>
            <th>Compress Time (ms)</th>
            <th>Decompress Time (ms)</th>
            <th>Total Time (ms)</th>
            <th>Compress Throughput (MB/s)</th>
        </tr>
"""
        
        for result in algos:
            name = result.get('name', 'Unknown')
            comp_size = result.get('compressed_size', 0)
            ratio = result.get('compression_ratio', 0)
            space_saved = result.get('space_saved_percent', 0)
            comp_time = result.get('compress_time_ms', 0)
            decomp_time = result.get('decompress_time_ms', 0)
            total_time = result.get('total_time_ms', 0)
            throughput = result.get('throughput_compress_mbps', 0)
            
            ratio_style = ' class="best"' if ratio == max(a.get('compression_ratio', 0) for a in algos) else ''
            speed_style = ' class="best"' if comp_time == min(a.get('compress_time_ms', float('inf')) for a in algos) else ''
            
            html += f"""        <tr>
            <td>{name}</td>
            <td>{comp_size:,}</td>
            <td{ratio_style}>{ratio:.2f}x</td>
            <td>{space_saved:.1f}%</td>
            <td{speed_style}>{comp_time:.2f}</td>
            <td>{decomp_time:.2f}</td>
            <td>{total_time:.2f}</td>
            <td>{throughput:.2f}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        return str(output_path)


def main():
    """Main entry point."""
    # Find results file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_path = project_root / "compression_results" / "benchmark_results.json"
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run benchmark_compression.py first to generate results.")
        sys.exit(1)
    
    # Analyze
    analyzer = BenchmarkAnalyzer(str(results_path))
    
    analyzer.print_comparison_table()
    analyzer.print_ranking("compression_ratio", top_n=5)
    analyzer.print_ranking("compress_time_ms", top_n=5, ascending=True)
    analyzer.print_recommendations()
    
    # Generate HTML report
    html_path = analyzer.generate_html_report()
    print(f"\nHTML report generated: {html_path}")


if __name__ == "__main__":
    main()

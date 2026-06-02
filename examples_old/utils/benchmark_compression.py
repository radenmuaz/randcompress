"""
Benchmark compression algorithms on the Quran text dataset.
Tests multiple SOTA algorithms and logs results.
"""

import os
import sys
import time
import gzip
import bz2
import lzma
import json
from pathlib import Path
from typing import Dict, Tuple, List
import hashlib

# Try importing optional compression libraries
try:
    import zstandard as zstd
    HAS_ZSTANDARD = True
except ImportError:
    HAS_ZSTANDARD = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


class CompressionBenchmark:
    """Benchmark compression algorithms."""
    
    def __init__(self, data_path: str, output_dir: str = "compression_results"):
        """
        Initialize benchmark.
        
        Args:
            data_path: Path to file to compress
            output_dir: Directory to save results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        with open(self.data_path, 'rb') as f:
            self.data = f.read()
        
        self.original_size = len(self.data)
        self.results = {}
        
        print(f"Loaded data: {self.data_path}")
        print(f"Original size: {self.original_size:,} bytes ({self.original_size / (1024**2):.2f} MB)")
        print(f"Output directory: {self.output_dir}\n")
    
    def _measure_algorithm(
        self,
        name: str,
        compress_fn,
        decompress_fn,
        iterations: int = 3
    ) -> Dict:
        """
        Measure compression/decompression performance.
        
        Args:
            name: Algorithm name
            compress_fn: Compression function
            decompress_fn: Decompression function
            iterations: Number of iterations for timing
            
        Returns:
            Dictionary with metrics
        """
        print(f"Testing {name}...", end=" ", flush=True)
        
        try:
            # Compression timing
            compress_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compressed = compress_fn(self.data)
                compress_times.append(time.perf_counter() - start)
            
            compress_time = min(compress_times)
            
            # Decompression timing
            decompress_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                decompressed = decompress_fn(compressed)
                decompress_times.append(time.perf_counter() - start)
            
            decompress_time = min(decompress_times)
            
            # Verify correctness
            if decompressed != self.data:
                raise ValueError("Decompressed data doesn't match original")
            
            compressed_size = len(compressed)
            ratio = self.original_size / compressed_size
            
            result = {
                "name": name,
                "original_size": self.original_size,
                "compressed_size": compressed_size,
                "compression_ratio": ratio,
                "compress_time_ms": compress_time * 1000,
                "decompress_time_ms": decompress_time * 1000,
                "total_time_ms": (compress_time + decompress_time) * 1000,
                "throughput_compress_mbps": (self.original_size / (1024**2)) / compress_time,
                "throughput_decompress_mbps": (compressed_size / (1024**2)) / decompress_time,
                "space_saved_percent": (1 - compressed_size / self.original_size) * 100,
            }
            
            print(f"✓ Ratio: {ratio:.2f}x, Compress: {result['compress_time_ms']:.2f}ms, "
                  f"Decompress: {result['decompress_time_ms']:.2f}ms")
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return {"name": name, "error": str(e)}
    
    def benchmark_gzip(self):
        """Benchmark gzip."""
        def compress(data):
            return gzip.compress(data, compresslevel=9)
        
        def decompress(data):
            return gzip.decompress(data)
        
        return self._measure_algorithm("gzip (level 9)", compress, decompress)
    
    def benchmark_bzip2(self):
        """Benchmark bzip2."""
        def compress(data):
            return bz2.compress(data, compresslevel=9)
        
        def decompress(data):
            return bz2.decompress(data)
        
        return self._measure_algorithm("bzip2 (level 9)", compress, decompress)
    
    def benchmark_lzma(self):
        """Benchmark lzma (xz)."""
        def compress(data):
            return lzma.compress(data, preset=9)
        
        def decompress(data):
            return lzma.decompress(data)
        
        return self._measure_algorithm("lzma/xz (preset 9)", compress, decompress)
    
    def benchmark_zstandard(self):
        """Benchmark zstandard."""
        if not HAS_ZSTANDARD:
            print("zstandard: ✗ Not installed (pip install zstandard)")
            return {"name": "zstandard", "error": "Not installed"}
        
        def compress(data):
            cctx = zstd.ZstdCompressor(level=22)
            return cctx.compress(data)
        
        def decompress(data):
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        
        return self._measure_algorithm("zstandard (level 22)", compress, decompress)
    
    def benchmark_lz4(self):
        """Benchmark LZ4."""
        if not HAS_LZ4:
            print("LZ4: ✗ Not installed (pip install lz4)")
            return {"name": "lz4", "error": "Not installed"}
        
        def compress(data):
            return lz4.frame.compress(data, compression_level=12)
        
        def decompress(data):
            return lz4.frame.decompress(data)
        
        return self._measure_algorithm("LZ4 (level 12)", compress, decompress)
    
    def benchmark_store(self):
        """Benchmark store (no compression)."""
        def compress(data):
            return data
        
        def decompress(data):
            return data
        
        return self._measure_algorithm("store (no compression)", compress, decompress)
    
    def run_all(self) -> Dict:
        """Run all benchmarks."""
        print("=" * 80)
        print("COMPRESSION ALGORITHM BENCHMARKS")
        print("=" * 80 + "\n")
        
        benchmarks = [
            self.benchmark_store,
            self.benchmark_gzip,
            self.benchmark_bzip2,
            self.benchmark_lzma,
            self.benchmark_zstandard,
            self.benchmark_lz4,
        ]
        
        results = []
        for benchmark_fn in benchmarks:
            result = benchmark_fn()
            results.append(result)
        
        self.results = {
            "file": str(self.data_path),
            "original_size": self.original_size,
            "algorithms": results,
        }
        
        return self.results
    
    def save_results(self, format: str = "json") -> Path:
        """
        Save results to file.
        
        Args:
            format: Output format ("json" or "csv")
            
        Returns:
            Path to saved file
        """
        if format == "json":
            output_file = self.output_dir / "benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
            return output_file
        
        elif format == "csv":
            import csv
            output_file = self.output_dir / "benchmark_results.csv"
            
            with open(output_file, 'w', newline='') as f:
                if not self.results.get("algorithms"):
                    return output_file
                
                # Get all possible keys from first result with algorithm data
                fieldnames = [
                    "name",
                    "compressed_size",
                    "compression_ratio",
                    "compress_time_ms",
                    "decompress_time_ms",
                    "total_time_ms",
                    "throughput_compress_mbps",
                    "throughput_decompress_mbps",
                    "space_saved_percent",
                    "error",
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for algo in self.results["algorithms"]:
                    # Only include specified fields
                    row = {k: algo.get(k, "") for k in fieldnames}
                    writer.writerow(row)
            
            print(f"Results saved to: {output_file}")
            return output_file
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def print_summary(self):
        """Print summary table."""
        print("\n" + "=" * 130)
        print("SUMMARY")
        print("=" * 130)
        
        if not self.results.get("algorithms"):
            print("No results to summarize")
            return
        
        # Print table header
        print(f"{'Algorithm':<30} {'Ratio':>12} {'Compress Time':>18} {'Decompress Time':>18} {'Space Saved':>15}")
        print("-" * 130)
        
        # Sort by compression ratio
        algos = [a for a in self.results["algorithms"] if "error" not in a or not a["error"]]
        algos.sort(key=lambda x: x.get("compression_ratio", 0), reverse=True)
        
        for result in algos:
            if "error" in result and result["error"]:
                print(f"{result['name']:<30} {'N/A':>12}")
            else:
                ratio_str = f"{result.get('compression_ratio', 0):.2f}x"
                comp_time_str = f"{result.get('compress_time_ms', 0):.2f}ms"
                decomp_time_str = f"{result.get('decompress_time_ms', 0):.2f}ms"
                space_saved_str = f"{result.get('space_saved_percent', 0):.1f}%"
                
                print(
                    f"{result['name']:<30} "
                    f"{ratio_str:>12} "
                    f"{comp_time_str:>18} "
                    f"{decomp_time_str:>18} "
                    f"{space_saved_str:>15}"
                )
        
        print("=" * 130)


def main():
    """Main entry point."""
    # Find dataset
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / "datasets" / "quran-uthmani.txt"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Run benchmark
    benchmark = CompressionBenchmark(
        data_path=dataset_path,
        output_dir=project_root / "compression_results"
    )
    
    benchmark.run_all()
    benchmark.save_results("json")
    benchmark.save_results("csv")
    benchmark.print_summary()


if __name__ == "__main__":
    main()

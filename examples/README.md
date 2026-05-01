# Compression Benchmark Examples

This directory contains comprehensive benchmarking scripts for compression algorithms on the Quran dataset.

## Scripts

### 1. `run_benchmark.py` (Main Entry Point)
Quick-start script that runs the complete benchmark pipeline:
- Runs all compression algorithm benchmarks
- Saves results in JSON and CSV formats
- Generates analysis report and HTML visualization
- Prints detailed rankings and recommendations

**Usage:**
```bash
python run_benchmark.py
```

### 2. `benchmark_compression.py` 
Standalone benchmarking module with `CompressionBenchmark` class.

**Features:**
- Tests multiple SOTA compression algorithms:
  - **gzip** (level 9) - Standard Unix compression
  - **bzip2** (level 9) - Better compression ratio than gzip
  - **lzma/xz** (preset 9) - Highest compression ratio (slow)
  - **zstandard** (level 22) - Modern algorithm (optional, requires `pip install zstandard`)
  - **LZ4** (level 12) - Fast compression (optional, requires `pip install lz4`)
  - **store** - No compression (baseline)

- Metrics collected for each algorithm:
  - Compression time (milliseconds)
  - Decompression time (milliseconds)  
  - Compression ratio (original_size / compressed_size)
  - Compression throughput (MB/s)
  - Space saved (percentage)

- Automatic verification that decompressed data matches original

**Usage:**
```python
from benchmark_compression import CompressionBenchmark

benchmark = CompressionBenchmark("path/to/file.txt")
results = benchmark.run_all()
benchmark.save_results("json")
benchmark.save_results("csv")
benchmark.print_summary()
```

### 3. `analyze_results.py`
Analysis and visualization module with `BenchmarkAnalyzer` class.

**Features:**
- Load results from JSON
- Rank algorithms by any metric
- Print comprehensive comparison tables
- Generate recommendations (best ratio, best speed, best balanced)
- Generate HTML report with styling and highlighting

**Usage:**
```python
from analyze_results import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer("compression_results/benchmark_results.json")
analyzer.print_comparison_table()
analyzer.print_ranking("compression_ratio", top_n=5)
analyzer.print_recommendations()
html_path = analyzer.generate_html_report()
```

Or run directly:
```bash
python analyze_results.py
```

## Output Files

Results are saved in `../compression_results/`:

```
compression_results/
├── benchmark_results.json      # Detailed results in JSON format
├── benchmark_results.csv       # Results in CSV format for spreadsheet analysis
└── report.html                 # Interactive HTML report
```

## Installation of Optional Dependencies

For the full suite of algorithms, install optional compression libraries:

```bash
# Zstandard (modern compression from Meta/Facebook)
pip install zstandard

# LZ4 (fast compression)
pip install lz4
```

Without these, the benchmark will still run but skip those algorithms with a message.

## Example Output

Running the benchmark on the Quran dataset produces output like:

```
Testing store (no compression)... ✓ Ratio: 1.00x, Compress: 0.10ms, Decompress: 0.05ms
Testing gzip (level 9)... ✓ Ratio: 2.73x, Compress: 45.23ms, Decompress: 12.15ms
Testing bzip2 (level 9)... ✓ Ratio: 2.89x, Compress: 78.45ms, Decompress: 28.93ms
Testing lzma/xz (preset 9)... ✓ Ratio: 3.12x, Compress: 245.67ms, Decompress: 42.18ms
Testing zstandard (level 22)... ✓ Ratio: 3.05x, Compress: 144.23ms, Decompress: 15.67ms
Testing LZ4 (level 12)... ✓ Ratio: 2.15x, Compress: 8.91ms, Decompress: 3.45ms

============================================ SUMMARY ============================================
Algorithm                 Ratio         Compress       Decompress      Space Saved
------------------------------------------
lzma/xz (preset 9)        3.12x         245.67ms       42.18ms         67.9%
bzip2 (level 9)           2.89x         78.45ms        28.93ms         65.4%
zstandard (level 22)      3.05x         144.23ms       15.67ms         67.2%
gzip (level 9)            2.73x         45.23ms        12.15ms         63.3%
LZ4 (level 12)            2.15x         8.91ms         3.45ms          53.5%
store (no compression)    1.00x         0.10ms         0.05ms          0.0%
```

## Recommendations

- **Best compression ratio**: LZMA/XZ (good for archival)
- **Best speed**: LZ4 (good for real-time systems)
- **Best balanced**: Zstandard (good for most use cases)
- **Standard fallback**: gzip (widely supported)

## Customization

You can easily extend the benchmark by:

1. **Adding compression algorithms**: Add methods like `benchmark_algorithm()` to `CompressionBenchmark` class
2. **Changing compression levels**: Modify level parameters in each benchmark method
3. **Testing multiple files**: Run benchmark on different datasets
4. **Custom analysis**: Use `BenchmarkAnalyzer` with custom metrics

## Performance Notes

- Timing measurements use `time.perf_counter()` for high resolution
- Each algorithm is tested 3 times; minimum time is reported
- Decompression is verified to match original data
- All operations use the full file in memory for fair comparison
- LZMA is the slowest but achieves best compression for text
- LZ4 is fastest but with lower compression ratio

## See Also

- Main benchmark script: [benchmark_compression.py](benchmark_compression.py)
- Analysis script: [analyze_results.py](analyze_results.py)
- Original example: [1.py](1.py)

import time
import random
import statistics
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exmel.adt import _PianoMidi, PianoMidi, MelEvent

class PianoMidiComparativeBenchmark:
    """Benchmark class for comparing PianoMidi vs OptimizedPianoMidi performance."""
    
    def __init__(self):
        self.data_sizes = [100, 1000, 1_0000, 10_0000, 100_0000, 1000_0000]
        self.results = {}
        
    def generate_test_data(self, size: int) -> dict[int, list[tuple[float, int]]]:
        """
        Generate test data with specified number of events.
        
        Args:
            size: Number of events to generate
            
        Returns:
            dictionary mapping note numbers to lists of (time, velocity) tuples
        """
        events_by_note = {}
        
        # Generate events across multiple notes (60-72 for a typical octave)
        num_notes = min(13, size // 10 + 1)  # Ensure we have multiple notes
        notes = list(range(60, 60 + num_notes))
        
        # Distribute events across notes
        events_per_note = size // num_notes
        remaining_events = size % num_notes
        
        for i, note in enumerate(notes):
            note_events = events_per_note + (1 if i < remaining_events else 0)
            if note_events == 0:
                continue
                
            # Generate random times between 0 and 100 seconds
            times = sorted([random.uniform(0, 100) for _ in range(note_events)])
            velocities = [random.randint(40, 120) for _ in range(note_events)]
            
            events_by_note[note] = list(zip(times, velocities))
        
        return events_by_note
    
    def create_piano_midi_instances(self, events_by_note: dict[int, list[tuple[float, int]]]) -> tuple[_PianoMidi, PianoMidi]:
        """
        Create both PianoMidi and OptimizedPianoMidi instances with the given test data.
        
        Args:
            events_by_note: Test data dictionary
            
        Returns:
            tuple of (PianoMidi, OptimizedPianoMidi) instances with test data
        """
        # Create mock instances and populate them with test data
        class MockPianoMidi(_PianoMidi):
            def __init__(self, events_by_note: dict[int, list[tuple[float, int]]]):
                self.events_by_note = events_by_note
        
        class MockOptimizedPianoMidi(PianoMidi):
            def __init__(self, events_by_note: dict[int, list[tuple[float, int]]]):
                self.events_by_note = events_by_note
                self._build_optimized_indices()
        
        piano_midi = MockPianoMidi(events_by_note)
        optimized_piano_midi = MockOptimizedPianoMidi(events_by_note)
        
        return piano_midi, optimized_piano_midi
    
    def generate_query_events(self, events_by_note: dict[int, list[tuple[float, int]]], 
                            num_queries: int = 1000) -> list[MelEvent]:
        """
        Generate random query events for benchmarking.
        
        Args:
            events_by_note: The test data
            num_queries: Number of query events to generate
            
        Returns:
            list of MelEvent objects for querying
        """
        query_events = []
        
        # Get time range from the data
        all_times = []
        for events in events_by_note.values():
            all_times.extend([time for time, _ in events])
        
        min_time = min(all_times) if all_times else 0
        max_time = max(all_times) if all_times else 100
        
        # Generate random query events
        for _ in range(num_queries):
            time = random.uniform(min_time, max_time)
            note = random.choice(list(events_by_note.keys()))
            query_events.append(MelEvent(time, note))
        
        return query_events
    
    def benchmark_method(self, piano_midi_instance, query_events: list[MelEvent], 
                        method_name: str, **kwargs) -> list[float]:
        """
        Benchmark a specific method on a piano midi instance.
        
        Args:
            piano_midi_instance: PianoMidi or OptimizedPianoMidi instance to test
            query_events: list of query events
            method_name: Name of the method to benchmark
            **kwargs: Additional arguments for the method
            
        Returns:
            list of execution times in seconds
        """
        method = getattr(piano_midi_instance, method_name)
        times = []
        
        for query_event in query_events:
            start_time = time.perf_counter()
            try:
                result = method(query_event, **kwargs)
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                times.append(float('inf'))
                continue
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return times
    
    def run_comparative_benchmarks(self, num_queries: int = 1000, num_runs: int = 5):
        """
        Run comprehensive comparative benchmarks for both implementations.
        
        Args:
            num_queries: Number of queries per benchmark
            num_runs: Number of runs per benchmark for averaging
        """
        methods_to_test = [
            ('right_nearest', {}),
            ('left_nearest', {}),
            ('nearest', {}),
            ('right_nearest_multi', {'n': 5}),
            ('left_nearest_multi', {'n': 5}),
            ('nearest_multi', {'n': 5}),
        ]
        
        # Add OptimizedPianoMidi-specific methods
        optimized_only_methods = [
            ('nearest_global', {'n': 5}),
        ]
        
        print(f"Running comparative benchmarks with {num_queries} queries per test, {num_runs} runs per benchmark")
        print("=" * 100)
        
        for size in self.data_sizes:
            print(f"\nData size: {size:,} events")
            print("-" * 50)
            
            # Generate test data
            events_by_note = self.generate_test_data(size)
            piano_midi, optimized_piano_midi = self.create_piano_midi_instances(events_by_note)
            query_events = self.generate_query_events(events_by_note, num_queries)
            
            size_results = {
                'PianoMidi': {},
                'OptimizedPianoMidi': {}
            }
            
            # Test common methods on both implementations
            for method_name, kwargs in methods_to_test:
                print(f"  Testing {method_name}...")
                
                # Test original PianoMidi
                print(f"    PianoMidi: ", end="")
                all_times_original = []
                for run in range(num_runs):
                    times = self.benchmark_method(piano_midi, query_events, method_name, **kwargs)
                    all_times_original.extend(times)
                
                mean_time_original = statistics.mean(all_times_original)
                median_time_original = statistics.median(all_times_original)
                std_time_original = statistics.stdev(all_times_original) if len(all_times_original) > 1 else 0
                
                size_results['PianoMidi'][method_name] = {
                    'mean': mean_time_original,
                    'median': median_time_original,
                    'std': std_time_original,
                    'total_time': sum(all_times_original),
                    'queries_per_second': len(all_times_original) / sum(all_times_original)
                }
                
                print(f"Done {mean_time_original*1000:.3f}ms mean")
                
                # Test OptimizedPianoMidi
                print(f"    OptimizedPianoMidi: ", end="")
                all_times_optimized = []
                for run in range(num_runs):
                    times = self.benchmark_method(optimized_piano_midi, query_events, method_name, **kwargs)
                    all_times_optimized.extend(times)
                
                mean_time_optimized = statistics.mean(all_times_optimized)
                median_time_optimized = statistics.median(all_times_optimized)
                std_time_optimized = statistics.stdev(all_times_optimized) if len(all_times_optimized) > 1 else 0
                
                size_results['OptimizedPianoMidi'][method_name] = {
                    'mean': mean_time_optimized,
                    'median': median_time_optimized,
                    'std': std_time_optimized,
                    'total_time': sum(all_times_optimized),
                    'queries_per_second': len(all_times_optimized) / sum(all_times_optimized)
                }
                
                # Calculate speedup
                speedup = mean_time_original / mean_time_optimized if mean_time_optimized > 0 else float('inf')
                print(f"Done {mean_time_optimized*1000:.3f}ms mean ({speedup:.2f}x speedup)")
            
            # Test OptimizedPianoMidi-only methods
            for method_name, kwargs in optimized_only_methods:
                print(f"  Testing {method_name} (OptimizedPianoMidi only)...")
                
                all_times_optimized = []
                for run in range(num_runs):
                    times = self.benchmark_method(optimized_piano_midi, query_events, method_name, **kwargs)
                    all_times_optimized.extend(times)
                
                mean_time_optimized = statistics.mean(all_times_optimized)
                median_time_optimized = statistics.median(all_times_optimized)
                std_time_optimized = statistics.stdev(all_times_optimized) if len(all_times_optimized) > 1 else 0
                
                size_results['OptimizedPianoMidi'][method_name] = {
                    'mean': mean_time_optimized,
                    'median': median_time_optimized,
                    'std': std_time_optimized,
                    'total_time': sum(all_times_optimized),
                    'queries_per_second': len(all_times_optimized) / sum(all_times_optimized)
                }
                
                print(f"    OptimizedPianoMidi: Done {mean_time_optimized*1000:.3f}ms mean")
            
            self.results[size] = size_results
        
        print("\n" + "=" * 100)
        print("Comparative benchmark completed!")
    
    def print_comparative_summary(self):
        """Print a comparative summary of benchmark results."""
        print("\nCOMPARATIVE BENCHMARK SUMMARY")
        print("=" * 120)
        
        methods = ['right_nearest', 'left_nearest', 'nearest', 
                  'right_nearest_multi', 'left_nearest_multi', 'nearest_multi']
        
        # Print header
        print(f"{'Data Size':<12} {'Method':<20} {'PianoMidi (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
        print("-" * 120)
        
        # Print results
        for size in self.data_sizes:
            for method in methods:
                if (method in self.results[size]['PianoMidi'] and 
                    method in self.results[size]['OptimizedPianoMidi']):
                    
                    original_time = self.results[size]['PianoMidi'][method]['mean']
                    optimized_time = self.results[size]['OptimizedPianoMidi'][method]['mean']
                    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                    
                    print(f"{size:<12,} {method:<20} {original_time*1000:<15.3f} {optimized_time*1000:<15.3f} {speedup:<10.2f}x")
            
            # Print OptimizedPianoMidi-only methods
            if 'nearest_global' in self.results[size]['OptimizedPianoMidi']:
                optimized_time = self.results[size]['OptimizedPianoMidi']['nearest_global']['mean']
                print(f"{size:<12,} {'nearest_global':<20} {'N/A':<15} {optimized_time*1000:<15.3f} {'N/A':<10}")
            
            print()  # Empty line between data sizes
    
    def plot_comparative_results(self, save_path: str = "comparative_benchmark_results.png"):
        """
        Create comparative plots of benchmark results.
        
        Args:
            save_path: Path to save the plot
        """
        methods = ['right_nearest', 'left_nearest', 'nearest', 
                  'right_nearest_multi', 'left_nearest_multi', 'nearest_multi']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('PianoMidi vs OptimizedPianoMidi Performance Comparison', fontsize=16)
        
        for i, method in enumerate(methods):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            sizes = []
            original_means = []
            optimized_means = []
            
            for size in self.data_sizes:
                if (method in self.results[size]['PianoMidi'] and 
                    method in self.results[size]['OptimizedPianoMidi']):
                    sizes.append(size)
                    original_means.append(self.results[size]['PianoMidi'][method]['median'] * 1000)
                    optimized_means.append(self.results[size]['OptimizedPianoMidi'][method]['median'] * 1000)
            
            if sizes:
                ax.loglog(sizes, original_means, 'o-', label='PianoMidi', linewidth=2, markersize=6)
                ax.loglog(sizes, optimized_means, 's-', label='OptimizedPianoMidi', linewidth=2, markersize=6)
                ax.set_xlabel('Data Size (events)')
                ax.set_ylabel('Time (ms)')
                ax.set_title(f'{method.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparative plot saved to {save_path}")
    
    def export_comparative_results(self, filename: str = "comparative_benchmark_results.csv"):
        """
        Export comparative benchmark results to CSV.
        
        Args:
            filename: Output CSV filename
        """
        import csv
        
        methods = ['right_nearest', 'left_nearest', 'nearest', 
                  'right_nearest_multi', 'left_nearest_multi', 'nearest_multi']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Data_Size', 'Method', 'PianoMidi_mean_ms', 'OptimizedPianoMidi_mean_ms', 
                     'PianoMidi_median_ms', 'OptimizedPianoMidi_median_ms', 
                     'PianoMidi_queries_per_sec', 'OptimizedPianoMidi_queries_per_sec', 'Speedup']
            writer.writerow(header)
            
            # Write data
            for size in self.data_sizes:
                for method in methods:
                    if (method in self.results[size]['PianoMidi'] and 
                        method in self.results[size]['OptimizedPianoMidi']):
                        
                        original_result = self.results[size]['PianoMidi'][method]
                        optimized_result = self.results[size]['OptimizedPianoMidi'][method]
                        speedup = original_result['mean'] / optimized_result['mean'] if optimized_result['mean'] > 0 else float('inf')
                        
                        row = [
                            size,
                            method,
                            original_result['mean'] * 1000,
                            optimized_result['mean'] * 1000,
                            original_result['median'] * 1000,
                            optimized_result['median'] * 1000,
                            original_result['queries_per_second'],
                            optimized_result['queries_per_second'],
                            speedup
                        ]
                        writer.writerow(row)
                
                # Write OptimizedPianoMidi-only methods
                if 'nearest_global' in self.results[size]['OptimizedPianoMidi']:
                    optimized_result = self.results[size]['OptimizedPianoMidi']['nearest_global']
                    row = [
                        size,
                        'nearest_global',
                        'N/A',
                        optimized_result['mean'] * 1000,
                        'N/A',
                        optimized_result['median'] * 1000,
                        'N/A',
                        optimized_result['queries_per_second'],
                        'N/A'
                    ]
                    writer.writerow(row)
        
        print(f"\nComparative results exported to {filename}")


def main():
    """Main function to run the comparative benchmark."""
    print("PianoMidi vs OptimizedPianoMidi Comparative Performance Benchmark")
    print("=" * 80)
    
    # Create benchmark instance
    benchmark = PianoMidiComparativeBenchmark()
    
    # Run comparative benchmarks
    benchmark.run_comparative_benchmarks(num_queries=1000, num_runs=3)
    
    # Print summary
    benchmark.print_comparative_summary()
    
    # Create plots (if matplotlib is available)
    try:
        benchmark.plot_comparative_results()
    except ImportError:
        print("\nMatplotlib not available. Skipping plot generation.")
    
    # Export results
    benchmark.export_comparative_results()
    
    print("\nComparative benchmark completed successfully!")


if __name__ == "__main__":
    main() 
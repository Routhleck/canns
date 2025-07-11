import functools
import time


def benchmark(runs=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            fastest_time = float("inf")
            slowest_time = 0
            all_times = []

            print(f"--- Start Benchmark: '{func.__name__}' (Runs {runs} times) ---")

            result = func(*args, **kwargs)

            for _ in range(runs):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()

                run_time = end_time - start_time
                all_times.append(run_time)

                total_time += run_time
                if run_time < fastest_time:
                    fastest_time = run_time
                if run_time > slowest_time:
                    slowest_time = run_time

            average_time = total_time / runs

            print(f"--- Benchmark Results: '{func.__name__}' ---")
            print(f"Total Run times: {runs}")
            print(f"Total Times runs: {total_time:.6f} s")
            print(f"Average Time: {average_time:.6f} s")
            print(f"Fastest: {fastest_time:.6f} s")
            print(f"Slowest: {slowest_time:.6f} s")
            print("-" * 40)

            return result

        return wrapper

    return decorator

"""
Not used for coding the actual game, rather prints all developer information in terminal to aid in maintenance.
Currently prints: 
"""
import time

# Enable/disable all output for debugging. Off by default.
DEBUG = True

# specialized print function that only outputs when DEBUG is True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
    else:
        return

# Simple micro-profiler for timing code sections.
PROFILE = {}

def prof_start(name):
    """Mark the start of a profiled section."""
    if name not in PROFILE:
        PROFILE[name] = [0, 0.0]   # [calls, total_time]
    PROFILE[name][0] += 1
    return time.perf_counter()

def prof_end(name, t0):
    """Mark the end of a profiled section."""
    PROFILE[name][1] += (time.perf_counter() - t0)

def prof_report():
    """Print a summary of all profiled functions."""
    if DEBUG:
        print("\n==== MICRO PROFILER REPORT ====") # heading of table
        # Table contents: function name, calls, total time, average time per call
        for name, (calls, total) in PROFILE.items():
            avg = total / calls if calls else 0
            print(f"{name:25s}  calls={calls:8d}  total={total:8.4f}s  avg={avg:10.7f}s")
        print("===============================\n") # footer of table
    else:
        return
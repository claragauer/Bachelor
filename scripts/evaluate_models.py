from memory_profiler import memory_usage

def measure_memory_usage(func, *args, **kwargs):
    """
    Measures memory usage of a given function.
    
    Parameters:
        func (function): The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        result: The result of the function execution.
        mem_usage (list): A list containing the memory usage over time in MB.
    """
    mem_usage, result = memory_usage((func, args, kwargs), interval=0.1, retval=True)  # Measure memory usage
    peak_memory = max(mem_usage)  # Get the peak memory usage
    print(f"Peak memory usage of {func.__name__}: {peak_memory:.4f} MB")
    
    return result, peak_memory

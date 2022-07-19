# %%
import triton

def bench(functions, constructor, iter, x_names=['N']):
    names = []
    funcs = []
    for func in functions:
        if hasattr(func, "__getitem__"):
            names.append(func[1])
            funcs.append(func[0])
        elif hasattr(func, "__name__"):
            names.append(func.__name__)
            funcs.append(func)
        else: raise Exception('cannot get name')
    print(names)
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=x_names,  # argument names to use as an x-axis for the plot
            x_vals=iter,  # different possible values for `x_name`
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot
            line_vals=names,  # possible values for `line_arg``
            line_names=names,  # label name for the lines
            ylabel="GB/s",  # label name for the y-axis
            plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
            args={}
        )
    )
    def benchmark(N, provider):
        print(provider, N)
        inputs = list(constructor(N))
        for nm, fn in zip(names, funcs):
            if nm == provider:
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(*inputs))
                break
        n_elements = sum([t.nelement() for t in inputs if hasattr(t, 'nelement')])
        element_size = inputs[0].element_size()
        gbps = lambda ms: 2 * n_elements * element_size * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)


    benchmark.run(show_plots=True, print_data=True)

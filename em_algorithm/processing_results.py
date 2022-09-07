

def process_parallel_results(parallel_results):
    results = {
        "params": [],
        "iter_steps": [],
        "execution_time": []
    }
    for r in parallel_results:
        results["params"].append(r[0][0])
        results["iter_steps"].append(r[0][1])
        results["execution_time"].append(r[1])
    return results
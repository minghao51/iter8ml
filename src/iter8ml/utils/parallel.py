import os


def _effective_parallel_jobs(
    *, requested_jobs: int, n_tasks: int, n_samples: int, n_features: int
) -> int:
    if requested_jobs <= 1 or n_tasks <= 1:
        return 1
    cpu_cap = max(1, os.cpu_count() or 1)
    requested_cap = min(int(requested_jobs), n_tasks, cpu_cap)
    matrix_size = n_samples * n_features
    if matrix_size >= 1_000_000:
        return min(requested_cap, 2)
    return requested_cap

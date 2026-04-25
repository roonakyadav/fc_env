"""Local benchmark entry used by training validation and optional profiling."""

from __future__ import annotations

from kernel_sandbox import KernelSandboxEnv


def run_local_benchmark() -> float:
    env = KernelSandboxEnv()
    r = env.benchmark()
    return r.score


if __name__ == "__main__":
    s = run_local_benchmark()
    print("benchmark score:", s)

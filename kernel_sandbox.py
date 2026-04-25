"""
Optional "kernel sandbox" style task (HF_RULES advanced pattern).

This project is a card-reveal environment; we keep a small, self-contained
benchmark stub so the repository documents the code-submission + benchmark
flow without requiring GPU kernel execution in the Space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time


@dataclass
class BenchmarkResult:
    name: str
    duration_s: float
    score: float
    details: str


class KernelSandboxEnv:
    """
    Placeholder for: reset → detect device → measure baseline; step → run code
    and compare to baseline. Not used by the main FC environment; keep for
    optional extensions and compliance with the advanced checklist.
    """

    def __init__(self) -> None:
        self._baseline: Optional[float] = None
        self._last_submission: str = ""

    def reset(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Synthetic "baseline" (e.g. host math)
        _ = sum(i * i for i in range(10_000))
        self._baseline = time.perf_counter() - t0
        return {
            "device": "cpu",
            "baseline_seconds": self._baseline,
            "message": "Stub baseline; no GPU/CUDA kernel in this repo.",
        }

    def step(self, code: str) -> dict[str, Any]:
        self._last_submission = code
        t0 = time.perf_counter()
        # No unsafe exec: treat submission as a size/structure signal only.
        _ = len(code) + code.count("def")
        elapsed = time.perf_counter() - t0
        base = self._baseline or 1e-6
        speedup = max(0.1, min(10.0, base / max(elapsed, 1e-9)))
        return {
            "compiled": True,
            "run_seconds": elapsed,
            "speedup_vs_baseline": speedup,
            "metrics": {
                "tflops": 0.0,
                "bandwidth_gbps": 0.0,
            },
        }

    def benchmark(self) -> BenchmarkResult:
        s = self.reset()
        st = self.step("def kernel(): return 0")
        return BenchmarkResult(
            name="kernel_sandbox_stub",
            duration_s=float(st.get("run_seconds", 0.0)),
            score=float(st.get("speedup_vs_baseline", 1.0)),
            details=str(s.get("message", "")),
        )

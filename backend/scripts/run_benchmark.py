"""Run baseline vs streaming and all ablations; write JSON, CSV and Markdown
to data/benchmark/.

    python scripts/run_benchmark.py
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.benchmark.evaluator import run_all  # noqa: E402
from app.config import settings  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    results = asyncio.run(run_all(settings))
    for name, exp in results["experiments"].items():
        print(f"\n== {name}")
        print(json.dumps(exp["summary"], indent=1))
    print(f"\nWrote results to {settings.benchmark_dir} in {results['runtime_s']}s")

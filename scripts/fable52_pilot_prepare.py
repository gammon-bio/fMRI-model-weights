"""Fable 5.2 pilot (docs/fable52_pilot_plan.md): run the locked phase2_batch_prepare.py with the
fix-queue target 10 included / cap 20 prepared. The locked script itself is not modified.

Usage: .venv/bin/python scripts/fable52_pilot_prepare.py --out /home/user/phase2_episodes/fable52
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_batch_prepare as bp  # noqa: E402

bp.RULES["fix"] = {"target": 10, "cap": 20}
if __name__ == "__main__":
    sys.argv += ["--prefix", "fab", "--queues", "fix", "--workers", "1"]
    bp.main()

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_all_configs(pattern: str = "config*_*.yaml") -> None:
    configs = sorted(ROOT.glob(pattern))
    if not configs:
        raise SystemExit(f"No configs match pattern: {pattern}")

    for cfg in configs:
        print(f"\n=== Running {cfg.name} ===")
        cmd = ["python", str(ROOT / "main.py"), str(cfg)]
        # Use run(..., check=True) to stop on failures
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_all_configs()
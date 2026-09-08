import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent

steps = [
    ("ETL", "etl.py"),
    ("QUALITY", "quality_check.py"),
    ("TRANSFORM", "transform.py"),
    ("FORECAST", "forecast.py"),
    ("CLASSIFICATION", "classification.py")
]


def main():
    for name, script in steps:
        print(f"{name}...")

        result = subprocess.run(
            [sys.executable, str(WORK_DIR / script)],
            cwd=WORK_DIR
        )

        if result.returncode != 0:
            print(f"{name}: FAILED")
            sys.exit(result.returncode)

        print(f"{name}: SUCCESS")

    print("PIPELINE: SUCCESS")


if __name__ == "__main__":
    main()
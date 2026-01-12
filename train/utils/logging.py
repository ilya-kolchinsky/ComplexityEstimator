from pathlib import Path
from datetime import datetime
import json


class Logger:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "meta.txt").write_text(f"created: {datetime.now()}\n")

    def write_json(self, name: str, obj):
        (self.out / name).write_text(json.dumps(obj, indent=2))

    @staticmethod
    def print_every(step: int, every: int) -> bool:
        return step % every == 0

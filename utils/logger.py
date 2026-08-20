import csv
from pathlib import Path
class Logger:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def write(self,row):
        new=not self.path.exists()
        with self.path.open("a",newline="") as f:
            w=csv.DictWriter(f,row.keys())
            if new:w.writeheader()
            w.writerow(row)

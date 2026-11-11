"""Download public UCI data; extract only expected plain text files."""
import argparse
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"

def download(output):
    root = Path(output); root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "uci.zip"
        urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                name = Path(entry.filename).name
                if name.endswith(".txt") and not entry.is_dir():
                    destination = root / name
                    if destination.exists():
                        raise ValueError(f"refusing to overwrite {destination}")
                    with z.open(entry) as source, destination.open("wb") as target:
                        import shutil
                        shutil.copyfileobj(source,target)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="data/hydraulic")
    download(p.parse_args().output)

"""Download the BoWFire dataset (226 photos with fire masks) used as extra training data.

    python scripts/download_bowfire.py

Extracts to external/BoWFireDataset/dataset/{img,gt}. The images come from Flickr under
Creative Commons licenses; cite the dataset when publishing:
D. Y. T. Chino, L. P. S. Avalhais, J. F. Rodrigues and A. J. M. Traina, "BoWFire: Detection
of Fire in Still Images by Integrating Pixel Color and Texture Analysis", SIBGRAPI 2015.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bitbucket.org/gbdi/bowfire-dataset/downloads/BoWFireDataset.zip"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "external"
KEEP_PREFIXES = ("BoWFireDataset/dataset/img/", "BoWFireDataset/dataset/gt/", "BoWFireDataset/README.txt")


def main() -> None:
    dataset_dir = TARGET / "BoWFireDataset" / "dataset"
    if (dataset_dir / "img").is_dir() and (dataset_dir / "gt").is_dir():
        print(f"Already present: {dataset_dir}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bowfire.zip"
        print(f"Downloading {URL} (~290 MB)...")
        with urllib.request.urlopen(URL) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        with zipfile.ZipFile(archive) as zf:
            members = [
                n for n in zf.namelist() if n.startswith(KEEP_PREFIXES) and not n.endswith(".DS_Store")
            ]
            zf.extractall(TARGET, members)
    print(f"Extracted to {dataset_dir}")


if __name__ == "__main__":
    main()

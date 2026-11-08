"""
AAFV-Net Dataset Downloader
Downloads MORPH, FG-NET, and AgeDB-30/CALFW/CPLFW/LFW datasets from Kaggle
and updates configs/params.yaml with the actual downloaded paths.

Run this ONCE before any pipeline phase.

Usage (Jupyter / Kaggle notebook):
    %run src/download_datasets.py

Usage (terminal / PowerShell):
    python src/download_datasets.py

Usage (Kaggle notebook — data is already mounted, use --kaggle-mode):
    python src/download_datasets.py --kaggle-mode
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure kagglehub is installed
# ---------------------------------------------------------------------------
try:
    import kagglehub
except ImportError:
    print("kagglehub not found. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "kagglehub"],
                   check=True)
    import kagglehub

# ---------------------------------------------------------------------------
# Dataset handles
# ---------------------------------------------------------------------------
DATASETS = {
    "morph":     "chiragsaipanuganti/morph",
    "fgnet":     "mulukentesfaye/fgnet",
    "benchmark": "yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112",
}

# Known subpaths inside each downloaded dataset directory
# (inspect the cache after first download and update these if needed)
MORPH_IMAGES_SUBPATH   = "Dataset/Images"
MORPH_INDEX_SUBPATH    = "Dataset/Index"
FGNET_IMAGES_SUBPATH   = "FGNET/images"
BENCHMARK_SUBPATH      = "val"

# Kaggle-mounted paths (used when running inside a Kaggle notebook)
KAGGLE_MOUNTED = {
    "morph":     "/kaggle/input/morph",
    "fgnet":     "/kaggle/input/fgnet",
    "benchmark": "/kaggle/input/agedb-30-calfw-cplfw-lfw-aligned-112x112",
}
# Newer Kaggle notebooks mount under /kaggle/input/datasets/<owner>/<name>
for _k, _h in DATASETS.items():
    _alt = "/kaggle/input/datasets/" + _h
    if not os.path.exists(KAGGLE_MOUNTED[_k]) and os.path.exists(_alt):
        KAGGLE_MOUNTED[_k] = _alt


# ---------------------------------------------------------------------------
# Setup kaggle credentials
# ---------------------------------------------------------------------------
def setup_credentials():
    """
    Move kaggle.json from home dir to ~/.kaggle/kaggle.json if not already done.
    """
    src  = Path.home() / "kaggle.json"
    dest = Path.home() / ".kaggle" / "kaggle.json"

    if dest.exists():
        print(f"✓ Kaggle credentials already at: {dest}")
        return True

    if src.exists():
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        os.chmod(dest, 0o600)
        print(f"✓ Moved kaggle.json → {dest}")
        return True

    # Check env variables (CI / server environments)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✓ Using KAGGLE_USERNAME / KAGGLE_KEY environment variables")
        return True

    print("✗ No kaggle.json found at ~/kaggle.json and no env vars set.")
    print("  Go to https://www.kaggle.com/settings → API → 'Create New Token'")
    print("  Download kaggle.json and place it at:", Path.home() / "kaggle.json")
    print("  Then re-run this script.")
    return False


# ---------------------------------------------------------------------------
# Download one dataset
# ---------------------------------------------------------------------------
def download_dataset(handle: str, name: str) -> Path | None:
    print(f"\n[download] {name} ({handle}) ...")
    try:
        path = kagglehub.dataset_download(handle)
        path = Path(path)
        print(f"  ✓ Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Find images directory inside a downloaded dataset root
# ---------------------------------------------------------------------------
def find_subpath(root: Path, subpath: str) -> Path:
    """
    Try the known subpath first; fall back to the root itself.
    Prints a warning if the subpath doesn't exist.
    """
    target = root / subpath
    if target.exists():
        return target
    # Sometimes kagglehub nests an extra directory
    # Try one level deeper
    for child in root.iterdir():
        if child.is_dir():
            nested = child / subpath
            if nested.exists():
                print(f"  (found at nested path: {nested})")
                return nested
    print(f"  ⚠ Expected subpath '{subpath}' not found; using root: {root}")
    return root


# ---------------------------------------------------------------------------
# Update params.yaml with discovered paths
# ---------------------------------------------------------------------------
def update_params_yaml(ws: Path, updates: dict[str, str]):
    """
    Update specific keys in configs/params.yaml without touching anything else.
    """
    yaml_path = ws / "configs" / "params.yaml"
    if not yaml_path.exists():
        print(f"  ✗ params.yaml not found at {yaml_path}")
        return

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    changed = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, val in updates.items():
            if stripped.startswith(key + ":"):
                # Preserve any trailing comment
                comment = ""
                if "#" in line:
                    comment = "  " + line[line.index("#"):].strip()
                new_lines.append(f'{key}: "{val}"{comment}')
                changed.add(key)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    yaml_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    for key in sorted(changed):
        print(f"  ✓ params.yaml updated: {key} = {updates[key]}")


# ---------------------------------------------------------------------------
# Print directory tree (first 2 levels) for inspection
# ---------------------------------------------------------------------------
def show_tree(root: Path, label: str, max_depth: int = 2):
    print(f"\n[tree] {label}: {root}")
    if not root.exists():
        print("  (not found)")
        return
    count = 0
    for item in sorted(root.rglob("*")):
        depth = len(item.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * depth
        kind = "📁" if item.is_dir() else "📄"
        print(f"  {indent}{kind} {item.name}")
        count += 1
        if count > 40:
            print("  ... (truncated)")
            break


# ---------------------------------------------------------------------------
# Count files in a directory
# ---------------------------------------------------------------------------
def count_files(d: Path, exts: tuple = (".jpg", ".png", ".JPG", ".PNG")) -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.rglob("*") if f.suffix in exts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download AAFV-Net datasets from Kaggle")
    parser.add_argument("--kaggle-mode", action="store_true",
                        help="Running inside a Kaggle notebook (data already mounted)")
    parser.add_argument("--skip-creds", action="store_true",
                        help="Skip credential check (already configured)")
    parser.add_argument("--only", nargs="*", choices=["morph", "fgnet", "benchmark"],
                        help="Download only the listed datasets")
    args = parser.parse_args()

    # Workspace root = parent of src/
    ws = Path(__file__).resolve().parent.parent
    print(f"Workspace: {ws}")

    # ---- Kaggle-mounted mode (running inside Kaggle) ----
    if args.kaggle_mode:
        print("\n[Kaggle-mounted mode] Using pre-mounted paths.")
        updates = {}
        for name, mnt_root in KAGGLE_MOUNTED.items():
            mnt_path = Path(mnt_root)
            if mnt_path.exists():
                print(f"  ✓ {name}: {mnt_path}")
            else:
                print(f"  ✗ {name}: {mnt_path} not found — check dataset name spelling")

        morph_img  = Path(KAGGLE_MOUNTED["morph"])  / MORPH_IMAGES_SUBPATH
        morph_idx  = Path(KAGGLE_MOUNTED["morph"])  / MORPH_INDEX_SUBPATH
        fgnet_img  = Path(KAGGLE_MOUNTED["fgnet"])  / FGNET_IMAGES_SUBPATH
        bench_dir  = Path(KAGGLE_MOUNTED["benchmark"]) / BENCHMARK_SUBPATH

        update_params_yaml(ws, {
            "MORPH_IMAGES_DIR":  str(morph_img),
            "MORPH_INDEX_DIR":   str(morph_idx),
            "FGNET_IMAGES_DIR":  str(fgnet_img),
            "BENCHMARK_DIR":     str(bench_dir),
        })
        print("\n✓ params.yaml updated for Kaggle-mounted paths.")
        return

    # ---- Regular download mode ----
    if not args.skip_creds:
        ok = setup_credentials()
        if not ok:
            sys.exit(1)

    to_download = args.only or list(DATASETS.keys())
    downloaded: dict[str, Path | None] = {}

    for name in to_download:
        handle = DATASETS[name]
        downloaded[name] = download_dataset(handle, name)

    # ---- Resolve subpaths ----
    paths_found: dict[str, str] = {}

    if downloaded.get("morph"):
        morph_root = downloaded["morph"]
        morph_img  = find_subpath(morph_root, MORPH_IMAGES_SUBPATH)
        morph_idx  = find_subpath(morph_root, MORPH_INDEX_SUBPATH)
        paths_found["MORPH_IMAGES_DIR"] = str(morph_img)
        paths_found["MORPH_INDEX_DIR"]  = str(morph_idx)
        show_tree(morph_root, "MORPH root")
        n_morph = count_files(morph_img)
        print(f"  Image count in {morph_img}: {n_morph}")

    if downloaded.get("fgnet"):
        fgnet_root = downloaded["fgnet"]
        fgnet_img  = find_subpath(fgnet_root, FGNET_IMAGES_SUBPATH)
        paths_found["FGNET_IMAGES_DIR"] = str(fgnet_img)
        show_tree(fgnet_root, "FG-NET root")
        n_fgnet = count_files(fgnet_img)
        print(f"  Image count in {fgnet_img}: {n_fgnet}")

    if downloaded.get("benchmark"):
        bench_root = downloaded["benchmark"]
        bench_dir  = find_subpath(bench_root, BENCHMARK_SUBPATH)
        paths_found["BENCHMARK_DIR"] = str(bench_dir)
        show_tree(bench_root, "Benchmark root")

    # ---- Update params.yaml ----
    if paths_found:
        print("\n[update] Writing discovered paths to params.yaml ...")
        update_params_yaml(ws, paths_found)
    else:
        print("\n⚠ No paths discovered (all downloads may have failed).")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for key, val in paths_found.items():
        p = Path(val)
        status = "✓ exists" if p.exists() else "✗ MISSING"
        print(f"  {key}: {val}  [{status}]")

    if not paths_found:
        print("  Nothing downloaded. Check errors above.")
        sys.exit(1)

    print("\nNext step:")
    print("  python src/run_pipeline.py --phases E0 P1 P2 P3 --dry-run")


if __name__ == "__main__":
    main()

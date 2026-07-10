"""Benchmark + equivalence harness for issue #47 (avoid loading all timestamps
into memory).

Two things, run together:

1. **Baseline + regression of speed and memory.** Runs the real end-to-end
   conversion on the bundled sample data and reports wall time and peak RSS, plus
   a synthetic at-scale probe of the `sample_count` materialisation (the
   conflict-free #47 surface in ``convert_intervals.add_sample_count``) so the
   memory win is visible even though the sample data is tiny.

2. **"Don't change the answer" guard.** Fingerprints the key datasets of the
   produced NWB (sha1 of the raw bytes). The first run writes
   ``bench_47_baseline.json``; later runs compare against it and FAIL loudly if
   any dataset changed. Any #47 refactor must keep every fingerprint identical.

Usage::

    python benchmarks/bench_47_timestamp_memory.py            # run + compare (or seed baseline)
    python benchmarks/bench_47_timestamp_memory.py --reseed   # overwrite the baseline
    python benchmarks/bench_47_timestamp_memory.py --scale 1e8 --skip-full   # memory probe only

This is a developer tool, not part of the test suite (it is slow and its
absolute numbers are machine-dependent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "bench_47_baseline.json"
SECONDS_PER_HOUR = 3600
EPHYS_RATE_HZ = 30_000
REFERENCE_HOURS = 17.0  # the duration called out in issue #47


class PeakRSS:
    """Context manager sampling process RSS on a background thread.

    ``peak`` is the maximum resident set size (bytes) observed between enter and
    exit, and ``delta`` subtracts the RSS at entry so it reports the *extra*
    memory the wrapped work brought resident.
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = False
        self.start_rss = 0
        self.peak = 0

    def __enter__(self) -> "PeakRSS":
        self.start_rss = self._proc.memory_info().rss
        self.peak = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop:
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *_exc) -> None:
        self._stop = True
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def delta(self) -> int:
        return self.peak - self.start_rss


def _gib(n_bytes: float) -> float:
    return n_bytes / 1024**3


# --------------------------------------------------------------------------- #
# 1. Synthetic at-scale probe of the sample_count materialisation
# --------------------------------------------------------------------------- #
class _DiskBackedIO:
    """Stand-in for SpikeGadgetsRawIO that yields Trodes sample counts on demand
    (``np.arange``), the way the real reader streams them from the memmap rather
    than holding them resident."""

    def __init__(self, lo: int, hi: int):
        self._lo, self._hi = lo, hi
        # only shape[0] (the file's sample count) is read by the lazy path
        self._raw_memmap = np.empty((hi - lo, 0), dtype=np.uint8)

    def get_analogsignal_timestamps(self, i_start, i_stop):
        i_start = 0 if i_start is None else i_start
        i_stop = (self._hi - self._lo) if i_stop is None else i_stop
        return np.arange(self._lo + i_start, self._lo + i_stop, dtype=np.uint32)


def probe_sample_count_memory(n_samples: int, n_files: int = 2) -> dict:
    """Measure the *extra* resident memory the real ``add_sample_count`` brings.

    Calls the actual function (so it auto-reflects the refactor) with a rec_dci
    whose ``timestamps`` is the already-resident ephys array and whose Trodes
    sample counts are generated on demand (simulating the on-disk memmap). The
    measured delta is everything ``add_sample_count`` adds on top of the shared
    timestamps — i.e. exactly what #47 is about.
    """
    from datetime import datetime

    from pynwb import NWBFile

    from trodes_to_nwb.convert_intervals import add_sample_count

    shared_timestamps = np.linspace(
        0.0, n_samples / EPHYS_RATE_HZ, n_samples, dtype=np.float64
    )
    edges = np.linspace(0, n_samples, n_files + 1, dtype=np.int64)

    class _RecDci:
        timestamps = shared_timestamps
        neo_io = [
            _DiskBackedIO(int(edges[i]), int(edges[i + 1])) for i in range(n_files)
        ]

    nwbfile = NWBFile(
        session_description="bench",
        identifier="bench",
        session_start_time=datetime(2023, 1, 1),
    )
    with PeakRSS() as rss:
        start = time.perf_counter()
        add_sample_count(nwbfile, _RecDci())
        wall = time.perf_counter() - start

    extra = rss.delta
    return {
        "n_samples": n_samples,
        "wall_s": wall,
        "extra_rss_bytes": extra,
        "extra_rss_gib": _gib(extra),
    }


# --------------------------------------------------------------------------- #
# 2. End-to-end conversion: speed, memory, and output fingerprint
# --------------------------------------------------------------------------- #
def _fingerprint_array(arr) -> dict:
    a = np.ascontiguousarray(arr[:])
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha1": hashlib.sha1(a.tobytes()).hexdigest(),
    }


def fingerprint_nwb(path: Path) -> dict:
    """Hash the datasets a #47 timestamp refactor could plausibly perturb."""
    import pynwb

    fp: dict = {}
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        es = nwb.acquisition["e-series"]
        fp["eseries.data"] = _fingerprint_array(es.data)
        fp["eseries.timestamps"] = _fingerprint_array(es.timestamps)
        if "sample_count" in nwb.processing:
            sc = nwb.processing["sample_count"]["sample_count"]
            fp["sample_count.data"] = _fingerprint_array(sc.data)
            fp["sample_count.timestamps"] = _fingerprint_array(sc.timestamps)
        if "analog" in nwb.processing:
            an = nwb.processing["analog"]["analog"]["analog"]
            fp["analog.data"] = _fingerprint_array(an.data)
            fp["analog.timestamps"] = _fingerprint_array(an.timestamps)
        if "behavior" in nwb.processing:
            be = nwb.processing["behavior"]["behavioral_events"]
            for name, ts in sorted(be.time_series.items()):
                fp[f"dio.{name}.data"] = _fingerprint_array(ts.data)
                fp[f"dio.{name}.timestamps"] = _fingerprint_array(ts.timestamps)
    return fp


def run_full_conversion() -> tuple[Path, dict]:
    """Run the bundled sample conversion (mirrors test_convert_full) under
    peak-RSS + wall-time measurement; returns the output path and metrics."""
    from trodes_to_nwb.convert import create_nwbs, get_included_device_metadata_paths
    from trodes_to_nwb.tests.utils import data_path

    device_metadata = get_included_device_metadata_paths()
    video_directory = data_path / "temp_video_directory_bench47"
    video_directory.mkdir(exist_ok=True)
    exclude = str(data_path / "20230622_sample_metadataProbeReconfig.yml")

    with PeakRSS() as rss:
        start = time.perf_counter()
        create_nwbs(
            path=data_path,
            device_metadata_paths=device_metadata,
            output_dir=str(data_path),
            n_workers=1,
            query_expression=f"animal == 'sample' and full_path != '{exclude}'",
            fs_gui_dir=data_path,
        )
        wall = time.perf_counter() - start

    output = data_path / "sample20230622.nwb"
    metrics = {
        "wall_s": wall,
        "peak_rss_bytes": rss.peak,
        "peak_rss_gib": _gib(rss.peak),
    }
    # tidy the report file + scratch dir; keep the nwb until after fingerprinting
    report = data_path / "sample20230622_nwbinspector_report.txt"
    if report.exists():
        report.unlink()
    shutil.rmtree(video_directory, ignore_errors=True)
    return output, metrics


def compare_fingerprints(baseline: dict, current: dict) -> list[str]:
    diffs = []
    for key in sorted(set(baseline) | set(current)):
        b, c = baseline.get(key), current.get(key)
        if b is None:
            diffs.append(f"  + {key}: present now, absent in baseline")
        elif c is None:
            diffs.append(f"  - {key}: in baseline, absent now")
        elif b != c:
            diffs.append(f"  ~ {key}: {b} -> {c}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scale",
        type=float,
        default=1e8,
        help="synthetic sample count for the memory probe (default 1e8)",
    )
    ap.add_argument(
        "--skip-full",
        action="store_true",
        help="skip the end-to-end conversion (memory probe only)",
    )
    ap.add_argument(
        "--reseed",
        action="store_true",
        help="overwrite the saved fingerprint/metrics baseline",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("Issue #47 benchmark: timestamp / sample_count memory")
    print("=" * 72)

    # --- synthetic at-scale memory probe -------------------------------------
    n = int(args.scale)
    probe = probe_sample_count_memory(n)
    factor = (REFERENCE_HOURS * SECONDS_PER_HOUR * EPHYS_RATE_HZ) / n
    print(f"\n[sample_count memory probe]  N = {n:,} samples")
    print(
        f"  add_sample_count extra RSS : {probe['extra_rss_gib']:.3f} GiB"
        f"  ({probe['wall_s'] * 1e3:.0f} ms)"
    )
    print(
        f"  extrapolated to {REFERENCE_HOURS:g} h @ {EPHYS_RATE_HZ // 1000} kHz "
        f"({factor:.1f}x): {probe['extra_rss_gib'] * factor:.1f} GiB"
    )

    result: dict = {"scale_probe": {k: v for k, v in probe.items() if k != "_held"}}

    # --- end-to-end conversion: speed, memory, equivalence -------------------
    if not args.skip_full:
        output, metrics = run_full_conversion()
        print(f"\n[full sample conversion]")
        print(f"  wall time : {metrics['wall_s']:.1f} s")
        print(f"  peak RSS  : {metrics['peak_rss_gib']:.3f} GiB")
        fp = fingerprint_nwb(output)
        result["full_metrics"] = metrics
        result["fingerprint"] = fp
        output.unlink(missing_ok=True)

        # Equivalence check vs saved baseline -- only when the baseline actually
        # carries a fingerprint. A fingerprint-less baseline (e.g. one seeded
        # under --skip-full) would otherwise compare against {} and flag every
        # dataset as "present now, absent in baseline", a false ANSWER CHANGED.
        base = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
        base_fp = base.get("fingerprint", {})
        if base_fp and not args.reseed:
            diffs = compare_fingerprints(base_fp, fp)
            print(f"\n[equivalence vs baseline]  ({len(fp)} datasets)")
            if diffs:
                print("  ANSWER CHANGED:")
                print("\n".join(diffs))
                return 1
            print("  OK - every dataset fingerprint matches the baseline.")
            bm = base.get("full_metrics", {})
            if bm:
                dt = metrics["wall_s"] - bm["wall_s"]
                dm = metrics["peak_rss_gib"] - bm["peak_rss_gib"]
                print(f"  speed  Δ {dt:+.1f} s (baseline {bm['wall_s']:.1f} s)")
                print(
                    f"  memory Δ {dm:+.3f} GiB (baseline {bm['peak_rss_gib']:.3f} GiB)"
                )
            return 0
        if base and not base_fp:
            print("\n[equivalence] baseline has no fingerprint; skipping (re-seed it).")

    # Seed the baseline only when a fingerprint was computed (i.e. the full
    # conversion ran). Seeding under --skip-full would write a fingerprint-less
    # baseline that poisons later equivalence checks (issue #47).
    if "fingerprint" in result and (not BASELINE_PATH.exists() or args.reseed):
        BASELINE_PATH.write_text(json.dumps(result, indent=2))
        print(f"\nSeeded baseline -> {BASELINE_PATH.name} (fingerprint included)")
    elif args.reseed:
        print("\n--reseed ignored: re-run without --skip-full to seed a baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

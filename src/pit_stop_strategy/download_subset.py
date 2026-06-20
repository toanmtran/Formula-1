"""Download FastF1 race data, one season at a time, with a hard per-race
timeout to prevent the FastF1 API stalling on a single timing-data call.

Resumable: already-present files are skipped.

Each race is loaded in a subprocess; if the subprocess does not finish in
RACE_TIMEOUT seconds, it is killed and the script moves on to the next race.
"""

import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import fastf1
import pandas as pd

from src.pit_stop_strategy.paths import PIPELINE_DATA_DIR, RAW_FASTF1_DIR

RACE_TIMEOUT = 240   # seconds per race; FastF1 timing-data sometimes hangs


def _load_race(year: int, round_no: int, gp: str, year_dir: Path,
               cache_dir: Path) -> None:
    """Worker run in its own process — entire FastF1 stack imported here."""
    import fastf1 as _ff1
    import pandas as _pd
    _ff1.Cache.enable_cache(str(cache_dir))

    session = _ff1.get_session(year, round_no, "R")
    load_telemetry = year >= 2018
    session.load(telemetry=load_telemetry)

    session.results.to_csv(year_dir / f"{gp}_Results.csv")
    session.laps.to_csv(year_dir / f"{gp}_Laps.csv")

    if load_telemetry:
        parts = []
        for driver in session.drivers:
            try:
                tel = session.laps.pick_drivers(driver).get_telemetry()
                tel["Driver"] = driver
                parts.append(tel)
            except Exception:
                pass
        if parts:
            _pd.concat(parts).to_csv(year_dir / f"{gp}_Telemetry.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2024])
    parser.add_argument("--max-races", type=int, default=None)
    args = parser.parse_args()

    cache_dir = PIPELINE_DATA_DIR / "f1_raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    csv_base_dir = RAW_FASTF1_DIR
    csv_base_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        year_dir = csv_base_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"!!! schedule load failed for {year}: {e}")
            continue

        done = 0
        for _, event in schedule.iterrows():
            if event["EventFormat"] == "testing":
                continue
            if args.max_races and done >= args.max_races:
                break

            gp = event["EventName"]
            tel_path = year_dir / f"{gp}_Telemetry.csv"
            results_path = year_dir / f"{gp}_Results.csv"
            laps_path = year_dir / f"{gp}_Laps.csv"
            if all(p.exists() for p in (tel_path, results_path, laps_path)):
                print(f"[skip] {year} {gp}")
                done += 1
                continue

            print(f"[load] {year} {gp}", flush=True)
            t_start = time.time()
            proc = mp.Process(
                target=_load_race,
                args=(year, int(event["RoundNumber"]), gp, year_dir, cache_dir),
                daemon=True,
            )
            proc.start()
            proc.join(timeout=RACE_TIMEOUT)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
                print(f"  TIMEOUT after {RACE_TIMEOUT}s on {year} {gp} "
                      f"— moving on", flush=True)
            else:
                elapsed = time.time() - t_start
                ok = tel_path.exists() or not (year >= 2018)
                tag = "OK" if ok else "PARTIAL"
                print(f"  {tag}  ({elapsed:.0f}s)", flush=True)

            done += 1
            time.sleep(2)

        print(f"\nFinished {year}: {done} race(s) attempted", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

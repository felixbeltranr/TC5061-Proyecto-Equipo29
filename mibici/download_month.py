from __future__ import annotations

import argparse

from mibici.dataset import download_month


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one MiBici monthly trips CSV into data/raw."
    )
    parser.add_argument("--year", type=int, required=True, help="Four-digit year.")
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    args = parser.parse_args()

    path = download_month(args.year, args.month)
    print(f"Downloaded {path.name} to {path.resolve()} ({path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()

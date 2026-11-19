#!/usr/bin/env python3
"""Fetch/verify pinned source without retaining a compressed download."""

from audit_reference import REFERENCE, snapshot


if __name__ == "__main__":
    record = snapshot()
    print(f"Verified {len(record['verified_files'])} reference files at {REFERENCE}")

"""Go Text Protocol entry point.

The command loop is a placeholder so the console script exists early. Fill this
out once the rules engine can answer protocol commands reliably.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Run a minimal GTP loop."""

    for raw_line in sys.stdin:
        command = raw_line.strip()
        if command in {"quit", "exit"}:
            print("=")
            print()
            return 0
        if command == "protocol_version":
            print("= 2")
        elif command == "name":
            print("= Sygo")
        elif command == "version":
            print("= 0.1.0")
        else:
            print(f"? unknown command: {command}")
        print()
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

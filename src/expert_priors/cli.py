"""Minimal command-line entry point: ``expert-priors``.

Currently exposes ``version``; subcommands (``elicit``, ``compare``) are added as
the corresponding library functions are ported.
"""

from __future__ import annotations

import argparse

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="expert-priors", description=__doc__)
    parser.add_argument("--version", action="version", version=f"expert-priors {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the installed version")

    args = parser.parse_args(argv)
    if args.command in (None, "version"):
        print(f"expert-priors {__version__}")
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

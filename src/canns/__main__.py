"""Command-line entry point for the CANNs toolkit.

`pip/uv install canns` installs several console scripts:
- `canns`: convenience wrapper (this module)
- `canns-tui`: Textual launcher (ASA / Gallery)
- `canns-gallery`: Gallery TUI
- `canns-gui`: ASA GUI (requires `canns[gui]`)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canns", description="CANNs toolkit entry point.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print installed CANNs version and exit.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--asa", action="store_true", help="Run the ASA TUI directly.")
    group.add_argument("--gallery", action="store_true", help="Run the model gallery TUI directly.")
    group.add_argument("--gui", action="store_true", help="Run the ASA GUI (requires canns[gui]).")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        try:
            from importlib.metadata import version

            print(version("canns"))
        except Exception:
            try:
                from canns._version import __version__

                print(__version__)
            except Exception:
                print("unknown")
        return 0

    if args.gui:
        from canns.pipeline.asa_gui import main as gui_main

        return int(gui_main())

    if args.gallery:
        from canns.pipeline.gallery import main as gallery_main

        gallery_main()
        return 0

    if args.asa:
        from canns.pipeline.asa import main as asa_main

        asa_main()
        return 0

    from canns.pipeline.launcher import main as launcher_main

    launcher_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

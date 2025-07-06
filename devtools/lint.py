import os.path
import subprocess
from pathlib import Path

from funlog import log_calls
from rich import get_console, reconfigure
from rich import print as rprint

# Update as needed.
DEVTOOLS_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).parent.parent

# 更新路径为相对于项目根目录
SRC_PATHS = [str(ROOT_DIR / "src")]
DOC_PATHS = [str(ROOT_DIR / "README.md")]

codespell_ignore = os.path.join(DEVTOOLS_DIR, "codespell_ignore.txt")

reconfigure(emoji=not get_console().options.legacy_windows)  # No emojis on legacy windows.


def main():
    rprint()

    errcount = 0
    errcount += run(["codespell", "--ignore-words", f"{codespell_ignore}", *SRC_PATHS, *DOC_PATHS])
    errcount += run(["ruff", "check", "--fix", *SRC_PATHS])
    errcount += run(["ruff", "format", *SRC_PATHS])
    # errcount += run(["basedpyright", "--stats", *SRC_PATHS])

    rprint()

    if errcount != 0:
        rprint(f"[bold red]:x: Lint failed with {errcount} errors.[/bold red]")
    else:
        rprint("[bold green]:white_check_mark: Lint passed![/bold green]")
    rprint()

    return errcount


@log_calls(level="warning", show_timing_only=True)
def run(cmd: list[str]) -> int:
    rprint()
    rprint(f"[bold green]>> {' '.join(cmd)}[/bold green]")
    errcount = 0
    try:
        subprocess.run(cmd, text=True, check=True)
    except KeyboardInterrupt:
        rprint("[yellow]Keyboard interrupt - Cancelled[/yellow]")
        errcount = 1
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error: {e}[/bold red]")
        errcount = 1

    return errcount


if __name__ == "__main__":
    exit(main())

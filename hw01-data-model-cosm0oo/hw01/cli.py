"""CLI utilities for homework variants."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CountResult:
    lines: int
    words: int
    bytes_count: int


def count_bytes(data: bytes) -> CountResult:
    return CountResult(
        lines=data.count(b"\n"),
        words=len(data.decode("utf-8", errors="replace").split()),
        bytes_count=len(data),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="my-wc", description="Count lines, words and bytes")
    parser.add_argument("-l", "--lines", action="store_true", help="Print line count")
    parser.add_argument("-w", "--words", action="store_true", help="Print word count")
    parser.add_argument(
        "-c",
        "--bytes",
        action="store_true",
        dest="bytes_mode",
        help="Print byte count",
    )
    parser.add_argument("files", nargs="*", help="Input files; if omitted, reads stdin")
    return parser


def format_output(
    result: CountResult,
    show_lines: bool,
    show_words: bool,
    show_bytes: bool,
    label: str | None,
) -> str:
    fields: list[str] = []
    if show_lines:
        fields.append(str(result.lines))
    if show_words:
        fields.append(str(result.words))
    if show_bytes:
        fields.append(str(result.bytes_count))
    if label:
        fields.append(label)
    return " ".join(fields)


def _read_file(path: str) -> bytes:
    return Path(path).read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    show_lines = args.lines
    show_words = args.words
    show_bytes = args.bytes_mode

    if not any([show_lines, show_words, show_bytes]):
        show_lines = True
        show_words = True
        show_bytes = True

    if not args.files:
        data = sys.stdin.buffer.read()
        result = count_bytes(data)
        print(format_output(result, show_lines, show_words, show_bytes, None))
        return 0

    exit_code = 0
    for file_path in args.files:
        try:
            data = _read_file(file_path)
        except FileNotFoundError:
            print(f"my-wc: {file_path}: No such file or directory", file=sys.stderr)
            exit_code = 1
            continue
        except PermissionError:
            print(f"my-wc: {file_path}: Permission denied", file=sys.stderr)
            exit_code = 1
            continue
        except IsADirectoryError:
            print(f"my-wc: {file_path}: Is a directory", file=sys.stderr)
            exit_code = 1
            continue
        except OSError as error:
            print(f"my-wc: {file_path}: {error.strerror}", file=sys.stderr)
            exit_code = 1
            continue

        result = count_bytes(data)
        print(format_output(result, show_lines, show_words, show_bytes, file_path))

    return exit_code

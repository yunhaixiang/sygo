"""Smart Game Format integration point."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sygo.engine.board import Color, Move, Point


@dataclass
class GameRecord:
    """A parsed or generated Go game record."""

    board_size: int = 19
    moves: list[Move] = field(default_factory=list)
    komi: float = 7.5
    rules: str = "CGOS/Tromp-Taylor"


def parse_sgf(content: str) -> GameRecord:
    """Parse SGF content.

    This intentionally supports the main-line subset Sygo needs now:
    root `SZ`/`KM` metadata and sequential `B[...]`/`W[...]` moves.
    Variations, setup stones, markup, and handicap records are left for a
    fuller SGF parser.
    """

    board_size = int(_first_property(content, "SZ") or "19")
    komi_value = _first_property(content, "KM")
    komi = float(komi_value) if komi_value is not None else 7.5
    rules = _first_property(content, "RU") or "CGOS/Tromp-Taylor"

    if board_size < 1 or board_size > 25:
        raise ValueError(f"Unsupported SGF board size: {board_size}")

    moves: list[Move] = []
    for color_text, value in re.findall(r";\s*([BW])\s*\[((?:\\.|[^\]])*)\]", content, re.I):
        color = Color.BLACK if color_text.upper() == "B" else Color.WHITE
        moves.append(Move(color=color, point=_parse_point(_unescape(value).strip(), board_size)))

    return GameRecord(board_size=board_size, moves=moves, komi=komi, rules=rules)


def write_sgf(record: GameRecord) -> str:
    """Serialize a game record as SGF."""

    properties = [
        "GM[1]",
        "FF[4]",
        "CA[UTF-8]",
        "AP[Sygo]",
        f"SZ[{record.board_size}]",
        f"KM[{record.komi:g}]",
        f"RU[{_escape(record.rules)}]",
    ]
    moves = "".join(
        f";{move.color.value[0].upper()}[{_format_point(move.point)}]" for move in record.moves
    )
    return f"(;{''.join(properties)}{moves})\n"


def _first_property(content: str, identifier: str) -> str | None:
    pattern = rf"{re.escape(identifier)}\s*\[((?:\\.|[^\]])*)\]"
    match = re.search(pattern, content, re.I)
    if match is None:
        return None
    return _unescape(match.group(1)).strip()


def _parse_point(value: str, board_size: int) -> Point | None:
    if value == "" or len(value) < 2:
        return None
    col = ord(value[0]) - ord("a")
    row = ord(value[1]) - ord("a")
    if not 0 <= row < board_size or not 0 <= col < board_size:
        raise ValueError(f"SGF point is outside {board_size}x{board_size}: {value}")
    return Point(row=row, col=col)


def _format_point(point: Point | None) -> str:
    if point is None:
        return ""
    return f"{chr(ord('a') + point.col)}{chr(ord('a') + point.row)}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]")


def _unescape(value: str) -> str:
    return re.sub(r"\\([\s\S])", r"\1", value)

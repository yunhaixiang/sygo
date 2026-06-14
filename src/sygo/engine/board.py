"""Core Go board primitives and 9x9-ready game state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sygo.engine.rules import DEFAULT_RULES, KoRule, Ruleset

EMPTY = 0
BLACK = 1
WHITE = 2
PASS_MOVE_INDEX_OFFSET = 1


class Color(str, Enum):
    """Stone colors used by the engine."""

    BLACK = "black"
    WHITE = "white"

    @property
    def opponent(self) -> "Color":
        return Color.WHITE if self is Color.BLACK else Color.BLACK

    @property
    def stone(self) -> int:
        return BLACK if self is Color.BLACK else WHITE

    @classmethod
    def from_stone(cls, stone: int) -> "Color":
        if stone == BLACK:
            return cls.BLACK
        if stone == WHITE:
            return cls.WHITE
        raise ValueError(f"Stone value does not map to a color: {stone}")


@dataclass(frozen=True)
class Point:
    """A zero-indexed board point."""

    row: int
    col: int


@dataclass
class Move:
    """A Go move. `point=None` represents pass."""

    color: Color
    point: Point | None


class IllegalMoveError(ValueError):
    """Raised when a move violates the configured rules."""


@dataclass
class GameState:
    """A compact Go game state.

    The implementation targets CGOS-style modified Tromp-Taylor defaults:
    area scoring, suicide forbidden, pass allowed, and simple ko initially.
    """

    size: int = 9
    board: tuple[tuple[int, ...], ...] | None = None
    to_play: Color = Color.BLACK
    rules: Ruleset = DEFAULT_RULES
    captures: dict[Color, int] = field(default_factory=lambda: {Color.BLACK: 0, Color.WHITE: 0})
    previous_board: tuple[tuple[int, ...], ...] | None = None
    consecutive_passes: int = 0
    move_count: int = 0

    def __post_init__(self) -> None:
        if self.board is None:
            self.board = tuple(tuple(EMPTY for _ in range(self.size)) for _ in range(self.size))
        if len(self.board) != self.size or any(len(row) != self.size for row in self.board):
            raise ValueError("Board dimensions must match size.")

    @property
    def is_over(self) -> bool:
        return self.consecutive_passes >= 2

    @property
    def pass_index(self) -> int:
        return self.size * self.size

    def move_to_index(self, point: Point | None) -> int:
        if point is None:
            return self.pass_index
        return point.row * self.size + point.col

    def index_to_point(self, index: int) -> Point | None:
        if index == self.pass_index:
            return None
        if index < 0 or index >= self.pass_index:
            raise ValueError(f"Move index out of range: {index}")
        return Point(row=index // self.size, col=index % self.size)

    def is_on_board(self, point: Point) -> bool:
        return 0 <= point.row < self.size and 0 <= point.col < self.size

    def get(self, point: Point) -> int:
        return self.board[point.row][point.col]

    def legal_moves(self) -> list[Point | None]:
        if self.is_over:
            return []
        moves: list[Point | None] = []
        for row in range(self.size):
            for col in range(self.size):
                point = Point(row, col)
                if self.is_legal(point):
                    moves.append(point)
        moves.append(None)
        return moves

    def legal_move_indices(self) -> list[int]:
        return [self.move_to_index(move) for move in self.legal_moves()]

    def is_legal(self, point: Point | None) -> bool:
        if self.is_over:
            return False
        if point is None:
            return True
        if not self.is_on_board(point) or self.get(point) != EMPTY:
            return False
        try:
            self.play(point)
        except IllegalMoveError:
            return False
        return True

    def play(self, point: Point | None) -> "GameState":
        """Return the next state after playing a move."""

        if self.is_over:
            raise IllegalMoveError("Cannot play after two consecutive passes.")
        if point is None:
            return self._pass()
        if not self.is_on_board(point):
            raise IllegalMoveError("Move is outside the board.")
        if self.get(point) != EMPTY:
            raise IllegalMoveError("Point is occupied.")

        board = [list(row) for row in self.board]
        color = self.to_play
        opponent = color.opponent
        board[point.row][point.col] = color.stone

        captured = 0
        for neighbor in self.neighbors(point):
            if board[neighbor.row][neighbor.col] != opponent.stone:
                continue
            group, liberties = self._collect_group_from_board(board, neighbor)
            if not liberties:
                captured += len(group)
                for stone in group:
                    board[stone.row][stone.col] = EMPTY

        own_group, own_liberties = self._collect_group_from_board(board, point)
        if not own_liberties and not self.rules.suicide_allowed:
            raise IllegalMoveError("Suicide is forbidden by the active ruleset.")
        if not own_liberties:
            for stone in own_group:
                board[stone.row][stone.col] = EMPTY

        next_board = tuple(tuple(row) for row in board)
        if self.rules.ko_rule is KoRule.SIMPLE and next_board == self.previous_board:
            raise IllegalMoveError("Simple ko violation.")

        captures = dict(self.captures)
        captures[color] += captured
        return GameState(
            size=self.size,
            board=next_board,
            to_play=opponent,
            rules=self.rules,
            captures=captures,
            previous_board=self.board,
            consecutive_passes=0,
            move_count=self.move_count + 1,
        )

    def area_score(self) -> float:
        """Return black score minus white score under simple area scoring."""

        black_area = 0
        white_area = self.rules.komi(self.size)
        ownership = self.ownership_map()

        for row in range(self.size):
            for col in range(self.size):
                owner = ownership[row][col]
                if owner == BLACK:
                    black_area += 1
                elif owner == WHITE:
                    white_area += 1

        return black_area - white_area

    def ownership_map(self) -> tuple[tuple[int, ...], ...]:
        """Return final area ownership as stone constants, or `EMPTY` for neutral points."""

        ownership = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        seen: set[Point] = set()

        for row in range(self.size):
            for col in range(self.size):
                point = Point(row, col)
                stone = self.get(point)
                if stone != EMPTY:
                    ownership[row][col] = stone
                elif point not in seen:
                    region, borders = self._collect_empty_region(point)
                    seen.update(region)
                    owner = EMPTY
                    if borders == {Color.BLACK}:
                        owner = BLACK
                    elif borders == {Color.WHITE}:
                        owner = WHITE
                    for region_point in region:
                        ownership[region_point.row][region_point.col] = owner

        return tuple(tuple(row) for row in ownership)

    def winner(self) -> Color | None:
        if not self.is_over:
            return None
        return Color.BLACK if self.area_score() > 0 else Color.WHITE

    def result_value(self, perspective: Color) -> float:
        """Return +1 for a win, -1 for a loss from `perspective`."""

        winner = self.winner()
        if winner is None:
            raise ValueError("Result value is only defined for finished games.")
        return 1.0 if winner is perspective else -1.0

    def neighbors(self, point: Point) -> list[Point]:
        candidates = (
            Point(point.row - 1, point.col),
            Point(point.row + 1, point.col),
            Point(point.row, point.col - 1),
            Point(point.row, point.col + 1),
        )
        return [candidate for candidate in candidates if self.is_on_board(candidate)]

    def _pass(self) -> "GameState":
        return GameState(
            size=self.size,
            board=self.board,
            to_play=self.to_play.opponent,
            rules=self.rules,
            captures=dict(self.captures),
            previous_board=self.board,
            consecutive_passes=self.consecutive_passes + 1,
            move_count=self.move_count + 1,
        )

    def _collect_empty_region(self, start: Point) -> tuple[set[Point], set[Color]]:
        region: set[Point] = set()
        borders: set[Color] = set()
        stack = [start]

        while stack:
            point = stack.pop()
            if point in region:
                continue
            region.add(point)
            for neighbor in self.neighbors(point):
                stone = self.get(neighbor)
                if stone == EMPTY and neighbor not in region:
                    stack.append(neighbor)
                elif stone != EMPTY:
                    borders.add(Color.from_stone(stone))

        return region, borders

    def _collect_group_from_board(
        self, board: list[list[int]], start: Point
    ) -> tuple[set[Point], set[Point]]:
        color = board[start.row][start.col]
        group: set[Point] = set()
        liberties: set[Point] = set()
        stack = [start]

        while stack:
            point = stack.pop()
            if point in group:
                continue
            group.add(point)
            for neighbor in self.neighbors(point):
                stone = board[neighbor.row][neighbor.col]
                if stone == EMPTY:
                    liberties.add(neighbor)
                elif stone == color and neighbor not in group:
                    stack.append(neighbor)

        return group, liberties

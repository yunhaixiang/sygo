from sygo.engine.board import Color, Move, Point
from sygo.engine.sgf import GameRecord, parse_sgf, write_sgf


def test_write_sgf_uses_standard_point_coordinates():
    record = GameRecord(
        board_size=9,
        komi=7.0,
        moves=[
            Move(Color.BLACK, Point(2, 3)),
            Move(Color.WHITE, None),
        ],
    )

    assert write_sgf(record) == (
        "(;GM[1]FF[4]CA[UTF-8]AP[Sygo]SZ[9]KM[7]RU[CGOS/Tromp-Taylor];B[dc];W[])\n"
    )


def test_parse_sgf_reads_main_line_moves():
    record = parse_sgf("(;GM[1]FF[4]SZ[9]KM[7]RU[CGOS/Tromp-Taylor];B[dc];W[];B[aa])")

    assert record.board_size == 9
    assert record.komi == 7.0
    assert record.moves == [
        Move(Color.BLACK, Point(2, 3)),
        Move(Color.WHITE, None),
        Move(Color.BLACK, Point(0, 0)),
    ]


def test_sgf_round_trip_preserves_simple_record():
    original = GameRecord(
        board_size=13,
        komi=7.5,
        moves=[
            Move(Color.BLACK, Point(6, 6)),
            Move(Color.WHITE, Point(5, 6)),
        ],
    )

    parsed = parse_sgf(write_sgf(original))

    assert parsed.board_size == original.board_size
    assert parsed.komi == original.komi
    assert parsed.moves == original.moves

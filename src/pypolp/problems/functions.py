from collections import namedtuple


Index = namedtuple('Index', 'start end')


def convert_to_namedtuples(indices: list[tuple[int, int]]) -> list[Index, ...]:
    return [Index(*item) for item in indices]


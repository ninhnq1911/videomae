from typing import List, Generator


def list_chunking(lst: List, n: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def custom_size_chunking(lst: List, frac_lst: List[float]) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from lst."""
    n = len(lst)
    offset = 0
    for frac in frac_lst:
        yield lst[offset : offset + int(n * frac)]
        offset = offset + int(n * frac)

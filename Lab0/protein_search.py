import csv
import gzip
import heapq
import sys
import time
from pathlib import Path


LETTERS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
K = 3
CANDIDATES = 300
TOP = 100


def parse(text):
    proteins = []
    title = None
    parts = []

    for row in text.splitlines():
        row = row.strip()
        if not row:
            continue
        if row.startswith(">"):
            if title is not None:
                proteins.append((title, "".join(parts).upper()))
            title = row[1:].strip()
            parts = []
        else:
            if title is None:
                raise ValueError("Первая строка должна начинаться с символа >")
            parts.append("".join(row.split()))

    if title is not None:
        proteins.append((title, "".join(parts).upper()))

    title, sequence = proteins[0]
    if not sequence:
        raise ValueError("Последовательность белка пустая")
    bad = sorted(set(sequence) - LETTERS)
    if bad:
        raise ValueError("Недопустимые символы: " + "".join(bad))

    return title, sequence


def read(path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    title = None
    parts = []

    with opener(path, "rt", encoding="ascii", errors="replace") as file:
        for row in file:
            row = row.strip()
            if not row:
                continue
            if row.startswith(">"):
                if title is not None:
                    yield title, "".join(parts).upper()
                title = row[1:].strip()
                parts = []
            else:
                parts.append(row)

    if title is not None:
        yield title, "".join(parts).upper()


def names(title):
    first = title.split(maxsplit=1)[0]
    parts = first.split("|")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return first, first


def words(sequence):
    return {sequence[i : i + K] for i in range(len(sequence) - K + 1)}


def jac(first, second):
    common = len(first & second)
    total = len(first) + len(second) - common
    return common / total if total else 0.0


def choose(database, query_title, query_sequence):
    query_words = words(query_sequence)
    query_id = names(query_title)[0]
    best = []
    started = time.perf_counter()

    for number, (title, sequence) in enumerate(read(database), 1):
        protein_id = names(title)[0]
        if protein_id == query_id:
            continue

        score = jac(query_words, words(sequence))
        item = (score, number, title, sequence)

        if len(best) < CANDIDATES:
            heapq.heappush(best, item)
        elif score > best[0][0]:
            heapq.heapreplace(best, item)

        if number % 100000 == 0:
            passed = time.perf_counter() - started
            print(f"Прочитано {number:,} белков за {passed:.1f} с", file=sys.stderr)

    return sorted(best, reverse=True)


def lev(first, second):
    if len(first) > len(second):
        first, second = second, first

    previous = list(range(len(first) + 1))

    for j, symbol2 in enumerate(second, 1):
        current = [j]
        left = j
        diagonal = j - 1

        for i, symbol1 in enumerate(first, 1):
            change = 0 if symbol1 == symbol2 else 1
            value = min(
                previous[i] + 1,
                left + 1,
                diagonal + change,
            )
            current.append(value)
            diagonal = previous[i]
            left = value

        previous = current

    return previous[-1]


def compare(query, candidates):
    result = []

    for jaccard, _, title, sequence in candidates:
        distance = lev(query, sequence)
        similarity = 1 - distance / max(len(query), len(sequence))
        protein_id, protein_name = names(title)
        result.append(
            (
                similarity,
                jaccard,
                distance,
                protein_id,
                protein_name,
                len(sequence),
                title,
            )
        )

    result.sort(reverse=True)
    return result[:TOP]


def save(path, result):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "accession",
                "entry",
                "length",
                "similarity",
                "distance",
                "jaccard",
                "header",
            ]
        )

        for rank, item in enumerate(result, 1):
            similarity, jaccard, distance, protein_id, protein_name, length, title = item
            writer.writerow(
                [
                    rank,
                    protein_id,
                    protein_name,
                    length,
                    f"{similarity:.6f}",
                    distance,
                    f"{jaccard:.6f}",
                    title,
                ]
            )


def main():
    database = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("results.csv")

    if sys.stdin.isatty():
        print("Вставь белок:")

    try:
        title, sequence = parse(sys.stdin.read())
        print(f"Запрос: {names(title)[0]}, длина: {len(sequence)}")

        start = time.perf_counter()
        candidates = choose(database, title, sequence)
        filter_time = time.perf_counter() - start
        print(f"Отбор 300 кандидатов: {filter_time:.2f} с")

        start = time.perf_counter()
        result = compare(sequence, candidates)
        compare_time = time.perf_counter() - start
        print(f"Левенштейн: {compare_time:.2f} с")

        save(output, result)
        print(f"Результат: {output.resolve()}")

        for rank, item in enumerate(result[:5], 1):
            similarity, jaccard, _, protein_id, protein_name, _, _ = item
            print(
                f"{rank}. {protein_id} {protein_name}: "
                f"similarity={similarity:.3f}, jaccard={jaccard:.3f}"
            )
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

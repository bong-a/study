"""Prepare the book's pinned TMDB corpus using only Python's standard library."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA_COMMIT = "ebb072430dde0acee8c327b05bfdba8fb1805c23"
BOOK_COMMIT = "cf09e6df739144bf798818f4af1366284a73a434"
BASE_URL = f"https://raw.githubusercontent.com/ai-powered-search/tmdb/{DATA_COMMIT}"
ARCHIVES = {
    "movies.tgz": "be491ff2681ca59ffbb1a2c729a7fceac7c1b8cb5f4c9b82dd50cec70f28a5d0",
    "judgments.tgz": "d699528ef320da3e33650219cb927ba2cd8b106453f0a5b3e99fcd42160cfe04",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name, cache):
    path = cache / name
    expected = ARCHIVES[name]
    if path.exists() and digest(path) == expected:
        print(f"Verified cache: {name}", flush=True)
        return path
    partial = cache / (name + ".download")
    print(f"Downloading {name} from the book's data repository...", flush=True)
    request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "ch10-opensearch-ltr-lab/1"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if digest(partial) != expected:
        raise ValueError(f"SHA-256 mismatch for {name}; refusing to use the downloaded data.")
    partial.replace(path)
    return path


def member_bytes(archive, name):
    # Read the one expected regular file; do not extract archive paths to disk.
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(name)
        if not member.isfile():
            raise ValueError(f"Expected a regular file: {name}")
        with bundle.extractfile(member) as stream:
            return stream.read()


def parse_judgments(text):
    queries, rows = {}, []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            header = re.fullmatch(r"#\s*qid:(\d+):\s*(.*?)\s*", line)
            if header:
                query = header.group(2)
                if "*" in query:
                    query, weight = query.rsplit("*", 1)
                    if weight != "1":
                        raise ValueError("This binary-judgment lab expects query weight 1.")
                queries[int(header.group(1))] = query
            continue
        match = re.fullmatch(r"([01])\s+qid:(\d+)\s+[^#]*#\s*(\S+).*", line)
        if not match:
            raise ValueError(f"Unrecognized judgment at line {number}: {line[:100]}")
        grade, qid, doc_id = match.groups()
        rows.append({"qid": int(qid), "doc_id": doc_id, "grade": int(grade)})
    seen, unique, duplicates = {}, [], []
    for row in rows:
        row["keywords"] = queries[row["qid"]]
        key = (row["qid"], row["doc_id"])
        if key in seen:
            if seen[key] != row["grade"]:
                raise ValueError(f"Conflicting judgments: {key}")
            duplicates.append(dict(row))
            continue
        seen[key] = row["grade"]
        unique.append(row)
    if not rows:
        raise ValueError("No judgments found.")
    return unique, duplicates


def prepare(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name, cache) for name in ARCHIVES}
    movies = json.loads(member_bytes(paths["movies.tgz"], "tmdb.json"))
    judgments, duplicates = parse_judgments(member_bytes(paths["judgments.tgz"], "ai_pow_search_judgments.txt").decode("utf-8"))
    missing_ids = sorted({row["doc_id"] for row in judgments} - set(movies))
    if missing_ids:
        raise ValueError(f"Judged documents missing from corpus: {missing_ids[:20]}")
    missing_years = 0
    movies_file = output / "movies.jsonl"
    with (output / "movies.jsonl.tmp").open("w", encoding="utf-8") as out:
        for doc_id, movie in sorted(movies.items(), key=lambda pair: int(pair[0])):
            release_date = movie.get("release_date") or ""
            valid_year = re.match(r"^(\d{4})-", release_date)
            year = int(valid_year[1]) if valid_year else 0
            missing_years += not bool(valid_year)
            document = {
                "id": str(doc_id),
                "title": movie.get("title") or "",
                "overview": movie.get("overview") or "",
                "release_year": year,
            }
            out.write(json.dumps(document, ensure_ascii=False) + "\n")
    (output / "movies.jsonl.tmp").replace(movies_file)
    (output / "judgments.json").write_text(json.dumps(judgments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "dataset": "AI-Powered Search / TMDB",
        "source_repository": "https://github.com/ai-powered-search/tmdb",
        "data_commit": DATA_COMMIT,
        "book_commit": BOOK_COMMIT,
        "archives": {name: {"url": f"{BASE_URL}/{name}", "sha256": sha} for name, sha in ARCHIVES.items()},
        "documents": len(movies), "judgments": len(judgments),
        "raw_judgment_rows": len(judgments) + len(duplicates),
        "duplicate_judgments_removed": duplicates,
        "queries": len({row["qid"] for row in judgments}),
        "grade_counts": dict(sorted(Counter(row["grade"] for row in judgments).items())),
        "documents_with_missing_release_year": missing_years,
        "transforms": [
            "Keep original document IDs, titles and overviews; null text becomes an empty string.",
            "Take the first four digits of release_date as integer release_year; missing dates become 0.",
            "Keep every source movie, including records the book's broader metadata loader may drop.",
            "Keep original qid, keywords, doc_id and binary grade; ignore any precomputed source features.",
            "Remove exact duplicate query/document/grade rows once; fail on contradictory labels and record every removed row.",
            "Recompute all feature scores in this OpenSearch index; do not reuse the book's Solr numbers.",
        ],
        "files": {"movies.jsonl": digest(movies_file), "judgments.json": digest(output / "judgments.json")},
        "attribution": "Movie metadata: The Movie Database (TMDB), as distributed by the book's data repository; educational use. See https://www.themoviedb.org/documentation/api/terms-of-use and the book repository's dataset notice.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: provenance[key] for key in ["documents", "judgments", "queries", "documents_with_missing_release_year"]}, indent=2))
    print(f"Prepared data in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        prepare(args.cache_dir, args.output_dir)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")

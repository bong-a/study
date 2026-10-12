#!/usr/bin/env python3
"""Chapter 10: log OpenSearch features, learn a pairwise SVM, and rerank movies.

Run from any directory. Data and artifacts live beside this file. No command
deletes an index, feature store, feature set, or model.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
INDEX = "ch10-movies"
FEATURESET = "ch10_movie_features"
FEATURE_NAMES = ("title_bm25", "overview_bm25", "release_year")
SEED = 42


def request(method, path, body=None, *, missing_ok=False, ndjson=False):
    """Return parsed JSON; only an explicitly allowed 404 becomes None."""
    base = os.environ.get("OPENSEARCH_URL", "http://localhost:19200").rstrip("/")
    data = body.encode("utf-8") if ndjson else (
        json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None
    )
    headers = {"Content-Type": "application/x-ndjson" if ndjson else "application/json"}
    req = Request(base + path, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as error:
        if error.code == 404 and missing_ok:
            return None
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {error.code}: {detail[:3000]}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"{method} {base + path}: {error}; check docker compose ps") from error


def save_json(name, value):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    # JSON strings may contain Unicode separators (U+0085/U+2028/U+2029).
    # Only literal LF bytes delimit JSONL records; str.splitlines() is broader.
    with path.open(encoding="utf-8", newline="\n") as source:
        return [json.loads(line) for line in source if line.strip()]


def checked_hits(response):
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError(f"Incomplete search response: {json.dumps(response)[:2000]}")
    return response["hits"]["hits"]


def doctor():
    info = request("GET", "/")
    health = request("GET", "/_cluster/health?wait_for_status=green&wait_for_nodes=2&timeout=30s")
    plugins = request("GET", "/_nodes/plugins")["nodes"]
    missing = [node.get("name", key) for key, node in plugins.items()
               if not any(plugin["name"] == "opensearch-ltr" for plugin in node["plugins"])]
    if missing:
        raise RuntimeError(f"opensearch-ltr missing on nodes: {missing}")
    if health.get("timed_out") or health["status"] != "green" or health["number_of_nodes"] != 2:
        raise RuntimeError(f"Expected a healthy two-node lab cluster: {health}")
    result = {"version": info["version"]["number"], "nodes": health["number_of_nodes"],
              "status": health["status"], "ltr_plugin": "present on every node"}
    if result["version"] != "3.3.2":
        raise RuntimeError(f"This lab is pinned to OpenSearch 3.3.2; found {result['version']}")
    save_json("doctor.json", result)
    print(json.dumps(result, indent=2))


def index_movies():
    movies = read_jsonl(DATA / "movies.jsonl")
    if not movies:
        raise ValueError("movies.jsonl is empty; run fetch_data.py first")
    ids = set()
    for movie in movies:
        if (not isinstance(movie.get("id"), str) or not movie["id"]
                or not isinstance(movie.get("title"), str)
                or not isinstance(movie.get("overview"), str)
                or type(movie.get("release_year")) is not int or movie["release_year"] < 0):
            raise ValueError(f"Invalid movie record: {movie!r}")
        if movie["id"] in ids:
            raise ValueError(f"Duplicate movie id: {movie['id']}")
        ids.add(movie["id"])
    mapping = {"properties": {
        "id": {"type": "keyword"},
        "title": {"type": "text", "analyzer": "standard"},
        "overview": {"type": "text", "analyzer": "standard"},
        "release_year": {"type": "integer"},
    }}
    existing = request("GET", f"/{INDEX}", missing_ok=True)
    if existing is None:
        request("PUT", f"/{INDEX}", {
            "settings": {"number_of_shards": 1, "number_of_replicas": 1}, "mappings": mapping,
        })
    else:
        settings = existing[INDEX]["settings"]["index"]
        properties = existing[INDEX]["mappings"].get("properties", {})
        if (str(settings["number_of_shards"]) != "1" or str(settings["number_of_replicas"]) != "1"
                or properties != mapping["properties"]):
            raise RuntimeError(f"Existing {INDEX} schema/settings differ from this lab; use a clean lab volume")
    for start in range(0, len(movies), 500):
        lines = []
        for movie in movies[start:start + 500]:
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": movie["id"]}}))
            lines.append(json.dumps(movie, ensure_ascii=False, allow_nan=False))
        result = request("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if result.get("errors"):
            errors = [item for item in result["items"] if item["index"].get("error")]
            raise RuntimeError(f"Bulk indexing failed: {errors[:3]}")
    request("POST", f"/{INDEX}/_refresh")
    count = request("GET", f"/{INDEX}/_count")["count"]
    if count != len(movies):
        raise RuntimeError(f"Expected {len(movies)} movies, found {count}; this lab index contains other documents")
    print(f"Indexed {count:,} movies into {INDEX} (1 primary, 1 replica).")


def feature_definitions():
    return [
        {"name": "title_bm25", "params": ["keywords"], "template_language": "mustache",
         "template": {"match": {"title": "{{keywords}}"}}},
        {"name": "overview_bm25", "params": ["keywords"], "template_language": "mustache",
         "template": {"match": {"overview": "{{keywords}}"}}},
        {"name": "release_year", "params": [], "template_language": "mustache",
         "template": {"function_score": {
             "query": {"match_all": {}},
             "field_value_factor": {"field": "release_year", "missing": 0},
             "boost_mode": "replace",
         }}},
    ]


def canonical_features(features):
    result = []
    for feature in features:
        template = feature["template"]
        if isinstance(template, str):
            template = json.loads(template)
        result.append({"name": feature["name"], "params": feature.get("params", []),
                       "template_language": feature.get("template_language", "mustache"),
                       "template": template})
    return result


def ensure_features():
    # Listing feature sets checks the default store itself, not just one missing set.
    if request("GET", "/_ltr/_featureset", missing_ok=True) is None:
        request("PUT", "/_ltr")
    expected = feature_definitions()
    payload = {"featureset": {"name": FEATURESET, "features": expected}}
    save_json("featureset_request.json", payload)
    existing = request("GET", f"/_ltr/_featureset/{FEATURESET}", missing_ok=True)
    if existing is None:
        request("POST", f"/_ltr/_featureset/{FEATURESET}", payload)
    else:
        actual = existing["_source"]["featureset"]["features"]
        if canonical_features(actual) != expected:
            raise RuntimeError(f"Existing {FEATURESET} differs; do not reuse logs/models from another feature definition")
    print(f"Feature set ready: {FEATURESET} ({', '.join(FEATURE_NAMES)})")


def log_request(query, doc_ids):
    return {
        "size": len(doc_ids), "_source": ["id", "title", "release_year"],
        "query": {"bool": {"filter": [
            {"ids": {"values": list(doc_ids)}},
            {"sltr": {"_name": "logged_features", "featureset": FEATURESET,
                      "params": {"keywords": query}}},
        ]}},
        "ext": {"ltr_log": {"log_specs": {
            "name": "features", "named_query": "logged_features", "missing_as_zero": True,
        }}},
    }


def extract_features(hit):
    entries = hit.get("fields", {}).get("_ltrlog", [])
    logs = [entry["features"] for entry in entries if "features" in entry]
    if len(logs) != 1:
        raise ValueError(f"Document {hit['_id']}: expected one _ltrlog.features entry")
    values = {}
    for feature in logs[0]:
        name = feature["name"]
        if name in values:
            raise ValueError(f"Document {hit['_id']}: duplicate feature {name}")
        value = feature.get("value")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Document {hit['_id']}: invalid feature {name}={value}")
        values[name] = float(value)
    if set(values) != set(FEATURE_NAMES):
        raise ValueError(f"Document {hit['_id']}: expected {FEATURE_NAMES}, got {tuple(values)}")
    return [values[name] for name in FEATURE_NAMES]


def fetch_features(query, doc_ids):
    """Filter by judged IDs so even text nonmatches are included with zero BM25."""
    doc_ids = list(doc_ids)
    result = {}
    for start in range(0, len(doc_ids), 500):
        batch = doc_ids[start:start + 500]
        hits = checked_hits(request("POST", f"/{INDEX}/_search?preference=ch10-ltr", log_request(query, batch)))
        result.update({hit["_id"]: extract_features(hit) for hit in hits})
    missing = set(doc_ids) - set(result)
    if missing:
        raise ValueError(f"Missing indexed documents for {query!r}: {sorted(missing)}")
    return result


def validate_rows(rows, *, with_features=False):
    if not rows:
        raise ValueError("No judgments to process")
    queries, seen = {}, set()
    for row in rows:
        if (type(row.get("qid")) is not int or not isinstance(row.get("keywords"), str)
                or not row["keywords"].strip() or not isinstance(row.get("doc_id"), str)
                or not row["doc_id"] or type(row.get("grade")) is not int or row["grade"] not in (0, 1)):
            raise ValueError(f"Invalid binary judgment: {row!r}")
        qid, doc_id = row["qid"], row["doc_id"]
        if qid in queries and queries[qid] != row["keywords"]:
            raise ValueError(f"qid {qid} has inconsistent keywords")
        queries[qid] = row["keywords"]
        if (qid, doc_id) in seen:
            raise ValueError(f"Duplicate judgment: qid {qid}, document {doc_id}")
        seen.add((qid, doc_id))
        if with_features:
            vector = row.get("features", [])
            if len(vector) != len(FEATURE_NAMES) or not np.isfinite(np.asarray(vector, dtype=float)).all():
                raise ValueError(f"Invalid feature vector for qid {qid}, document {doc_id}")


def log_judgments():
    rows = read_json(DATA / "judgments.json")
    validate_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    sample = next((judgments for judgments in grouped.values()
                   if judgments[0]["keywords"].casefold() == "social network"), next(iter(grouped.values())))
    save_json("log_request.json", log_request(sample[0]["keywords"], [row["doc_id"] for row in sample]))
    logged = []
    for qid, judgments in sorted(grouped.items()):
        query = judgments[0]["keywords"]
        values = fetch_features(query, [row["doc_id"] for row in judgments])
        logged.extend({**row, "features": values[row["doc_id"]]} for row in judgments)
        print(f"qid {qid}: {query!r}: logged {len(judgments)} judgments")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "logged_features.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in logged), encoding="utf-8"
    )
    print(f"Saved {len(logged):,} rows to artifacts/logged_features.jsonl")


def split_queries(rows):
    qids = sorted({row["qid"] for row in rows})
    if len(qids) < 2:
        raise ValueError("At least two query IDs are required for a held-out query split")
    train_qids, test_qids = train_test_split(qids, test_size=0.1, random_state=SEED)
    train_set = set(train_qids)
    return ([row for row in rows if row["qid"] in train_set],
            [row for row in rows if row["qid"] not in train_set])


def fit_scaler(train_rows):
    values = np.asarray([row["features"] for row in train_rows], dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return means, stds


def make_pairs(rows, means, stds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    differences, labels = [], []
    for judgments in grouped.values():
        for i, left in enumerate(judgments):
            for right in judgments[i + 1:]:
                if left["grade"] == right["grade"]:
                    continue
                # z_left - z_right = (x_left - x_right) / std: the train mean cancels.
                difference = (np.asarray(left["features"]) - np.asarray(right["features"])) / stds
                label = 1 if left["grade"] > right["grade"] else -1
                differences.extend([difference, -difference])
                labels.extend([label, -label])
    if not labels:
        raise ValueError("Training queries contain no unequal-grade pairs")
    return np.asarray(differences), np.asarray(labels)


def evaluate(rows, weights, means, stds, k=5):
    """Macro P@k over judged documents only; missing ranks contribute zero."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    results = []
    for qid, judgments in sorted(grouped.items()):
        def ltr_score(row):
            return float(((np.asarray(row["features"]) - means) / stds) @ weights)

        baseline = sorted(judgments, key=lambda row: (-sum(row["features"][:2]), row["doc_id"]))
        reranked = sorted(judgments, key=lambda row: (-ltr_score(row), row["doc_id"]))
        results.append({
            "qid": qid, "keywords": judgments[0]["keywords"], "judged_documents": len(judgments),
            "title_plus_overview_p_at_5": sum(row["grade"] > 0 for row in baseline[:k]) / k,
            "ltr_p_at_5": sum(row["grade"] > 0 for row in reranked[:k]) / k,
            "baseline_top_ids": [row["doc_id"] for row in baseline[:k]],
            "ltr_top_ids": [row["doc_id"] for row in reranked[:k]],
        })
    return {
        "scope": "held-out queries; judged documents only; macro average; relevance grade > 0",
        "k": k, "denominator": "k (unfilled ranks count as non-relevant)",
        "title_plus_overview_p_at_5": float(np.mean([row["title_plus_overview_p_at_5"] for row in results])),
        "ltr_p_at_5": float(np.mean([row["ltr_p_at_5"] for row in results])),
        "per_query": results,
    }


def model_payload(weights, means, stds):
    definition = {
        "type": "model/linear",
        "definition": {name: float(weights[i]) for i, name in enumerate(FEATURE_NAMES)},
        "feature_normalizers": {
            name: {"standard": {"mean": float(means[i]), "standard_deviation": float(stds[i])}}
            for i, name in enumerate(FEATURE_NAMES)
        },
    }
    digest = hashlib.sha256(json.dumps(
        {"model": definition, "features": feature_definitions()}, sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()[:12]
    return {"model": {"name": f"ch10_movie_svm_{digest}", "model": definition}}


def train_model():
    rows = read_jsonl(ARTIFACTS / "logged_features.jsonl")
    validate_rows(rows, with_features=True)
    train_rows, test_rows = split_queries(rows)
    means, stds = fit_scaler(train_rows)
    x_pairs, y_pairs = make_pairs(train_rows, means, stds)
    learner = LinearSVC(fit_intercept=False, dual="auto", max_iter=10000, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        learner.fit(x_pairs, y_pairs)
    weights = learner.coef_[0]
    payload = model_payload(weights, means, stds)
    summary = {
        "model_name": payload["model"]["name"], "feature_names": list(FEATURE_NAMES),
        "seed": SEED, "split": "query IDs: 90% train / 10% test; sklearn train_test_split",
        "train_qids": sorted({row["qid"] for row in train_rows}),
        "test_qids": sorted({row["qid"] for row in test_rows}),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "training_pairs": len(y_pairs), "pair_policy": "unequal grades within query; both directions",
        "means": means.tolist(), "stds": stds.tolist(), "weights": weights.tolist(),
        "normalization": "train rows only; population std (ddof=0); zero std replaced by 1",
        "learner": {"name": "LinearSVC", "fit_intercept": False, "dual": "auto", "max_iter": 10000},
        "test_metrics": evaluate(test_rows, weights, means, stds),
    }
    save_json("model.json", payload)
    save_json("training.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def canonical_model(model):
    result = dict(model)
    if isinstance(result.get("definition"), str):
        result["definition"] = json.loads(result["definition"])
    # OpenSearch stores only normalizer parameters as float32. Keep the type,
    # weights, and every dictionary key unchanged when checking model identity.
    result["feature_normalizers"] = {
        name: {kind: {key: float(np.float32(value)) for key, value in parameters.items()}
               for kind, parameters in normalizer.items()}
        for name, normalizer in result["feature_normalizers"].items()
    }
    return result


def upload_model():
    payload = read_json(ARTIFACTS / "model.json")
    name = payload["model"]["name"]
    existing = request("GET", f"/_ltr/_model/{name}", missing_ok=True)
    if existing is None:
        request("POST", f"/_ltr/_featureset/{FEATURESET}/_createmodel", payload)
        print(f"Uploaded immutable model: {name}")
    else:
        stored = existing["_source"]["model"]
        if (canonical_model(stored["model"]) != canonical_model(payload["model"]["model"])
                or canonical_features(stored["feature_set"]["features"]) != feature_definitions()):
            raise RuntimeError(f"Existing model {name} differs from the artifact; refusing to overwrite")
        print(f"Reused identical immutable model: {name}")


def search_request(query, size, window=None, model_name=None):
    if not query.strip() or size < 1 or size > 10000:
        raise ValueError("query must be nonempty and size must be between 1 and 10000")
    body = {
        "size": size, "track_total_hits": True, "_source": ["id", "title", "release_year"],
        "query": {"multi_match": {"query": query, "fields": ["title", "overview"], "type": "most_fields"}},
    }
    if model_name:
        if window is None or window < size or window > 10000:
            raise ValueError("rescore window must be >= size and <= 10000")
        body["rescore"] = {
            "window_size": window,
            "query": {"rescore_query": {"sltr": {"model": model_name, "params": {"keywords": query}}},
                      "query_weight": 0.0, "rescore_query_weight": 1.0, "score_mode": "total"},
        }
    return body


def predict(features, payload):
    definition = payload["model"]["model"]
    value = 0.0
    for i, name in enumerate(FEATURE_NAMES):
        normalizer = definition["feature_normalizers"][name]["standard"]
        value += ((features[i] - normalizer["mean"]) / normalizer["standard_deviation"]
                  * definition["definition"][name])
    return value


def search(query, window=500, size=5):
    payload = read_json(ARTIFACTS / "model.json")
    name = payload["model"]["name"]
    baseline_body = search_request(query, size)
    rerank_body = search_request(query, size, window, name)
    save_json("search_bm25_request.json", baseline_body)
    save_json("search_ltr_request.json", rerank_body)
    baseline = request("POST", f"/{INDEX}/_search?preference=ch10-ltr", baseline_body)
    reranked = request("POST", f"/{INDEX}/_search?preference=ch10-ltr", rerank_body)
    baseline_hits, reranked_hits = checked_hits(baseline), checked_hits(reranked)
    ids = [hit["_id"] for hit in reranked_hits]
    features = fetch_features(query, ids)
    comparison = []
    for hit in reranked_hits:
        predicted = predict(features[hit["_id"]], payload)
        actual = hit["_score"]
        if not math.isclose(predicted, actual, rel_tol=2e-5, abs_tol=2e-5):
            raise RuntimeError(f"Training/serving score mismatch for {hit['_id']}: Python {predicted}, OpenSearch {actual}")
        comparison.append({"doc_id": hit["_id"], "title": hit["_source"]["title"],
                           "release_year": hit["_source"]["release_year"], "features": features[hit["_id"]],
                           "opensearch_score": actual, "python_score": predicted,
                           "absolute_error": abs(predicted - actual)})
    result = {
        "query": query, "model_name": name, "window_size": window, "size": size,
        "matched_documents": baseline["hits"]["total"],
        "candidate_policy": "one primary shard; BM25 most_fields(title, overview), then top-window rescore",
        "baseline": [{"doc_id": hit["_id"], "title": hit["_source"]["title"], "score": hit["_score"]}
                     for hit in baseline_hits],
        "ltr": comparison, "score_parity": "passed" if comparison else "no matching documents",
        "requests": {"baseline": baseline_body, "ltr": rerank_body},
    }
    save_json("search_results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("doctor", "Check two green OpenSearch 3.3.2 nodes and the LTR plugin"),
        ("index", "Bulk upsert data/movies.jsonl into the dedicated lab index"),
        ("features", "Create or verify the three-feature set without deleting anything"),
        ("log", "Log all data/judgments.json rows to artifacts/logged_features.jsonl"),
        ("train", "Split query IDs 90/10, normalize train only, fit pairwise LinearSVC, evaluate"),
        ("upload", "Upload or reuse the content-addressed model in artifacts/model.json"),
        ("search", "Compare BM25 and LTR and check Python/OpenSearch score parity"),
        ("all", "Run doctor, index, features, log, train, upload, search in order"),
    ):
        command = commands.add_parser(name, help=help_text, description=help_text)
        if name in ("search", "all"):
            command.add_argument("--query", default="harry potter", help="Search text (default: harry potter)")
            command.add_argument("--window", type=int, default=500, help="Per-shard rescore candidates (default: 500)")
            command.add_argument("--size", type=int, default=5, help="Returned results, <= window (default: 5)")
    args = parser.parse_args(argv)
    actions = {"doctor": doctor, "index": index_movies, "features": ensure_features,
               "log": log_judgments, "train": train_model, "upload": upload_model}
    try:
        if args.command in ("all", "search"):
            search_request(args.query, args.size, args.window, "validate-input")
        if args.command == "all":
            for action in actions.values():
                action()
        elif args.command in actions:
            actions[args.command]()
        if args.command in ("all", "search"):
            search(args.query, args.window, args.size)
    except (RuntimeError, ValueError, OSError, ConvergenceWarning) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

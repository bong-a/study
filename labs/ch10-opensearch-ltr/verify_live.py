"""Optional end-to-end checks and example snapshots for the study guide."""
import json
import math

import lab


def main():
    lab.doctor()
    lab.ensure_features()
    lab.upload_model()
    examples = {}
    for query in ["social network", "star wars", "wrath of khan", "harry potter"]:
        lab.search(query)
        examples[query] = lab.read_json(lab.ARTIFACTS / "search_results.json")
    payload = lab.read_json(lab.ARTIFACTS / "model.json")
    query = "wrath of khan"
    ids = ["154", "157"]
    features = lab.fetch_features(query, ids)
    # Include both films explicitly, even if the initial text search would miss one.
    request = {
        "size": 2,
        "query": {"bool": {"filter": [{"ids": {"values": ids}}],
                           "must": [{"sltr": {"model": payload["model"]["name"],
                                                "params": {"keywords": query}}}]}},
    }
    hits = lab.checked_hits(lab.request("POST", f"/{lab.INDEX}/_search?preference=ch10-ltr", request))
    if len(hits) != 2:
        raise RuntimeError("Both Star Trek example documents must be present")
    star_trek = []
    for hit in hits:
        expected = lab.predict(features[hit["_id"]], payload)
        if not math.isclose(expected, hit["_score"], rel_tol=2e-5, abs_tol=2e-5):
            raise RuntimeError("Explicit-document score parity failed")
        star_trek.append({"id": hit["_id"], "title": hit["_source"]["title"],
                          "features": features[hit["_id"]], "python_score": expected,
                          "opensearch_score": hit["_score"]})
    training = lab.read_json(lab.ARTIFACTS / "training.json")
    if set(training["train_qids"]) & set(training["test_qids"]):
        raise RuntimeError("Train/test query leakage")
    result = {
        "passed": True,
        "storage_note": "Storage follows node startup: named volumes by default, disposable RAM with compose.ram.yaml.",
        "queries": examples,
        "explicit_document_example": {"query": query, "documents": star_trek, "request": request},
        "max_search_score_error": max(row["absolute_error"] for example in examples.values() for row in example["ltr"]),
        "model_reuse": "passed",
        "query_split_disjoint": True,
    }
    lab.save_json("verification.json", result)
    print(json.dumps({"passed": True, "queries": list(examples),
                      "max_search_score_error": result["max_search_score_error"],
                      "explicit_document_scores": star_trek}, indent=2))


if __name__ == "__main__":
    main()

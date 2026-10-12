"""Offline regression tests for the LTR data/learning/serving contract."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

SPEC = importlib.util.spec_from_file_location("lab", Path(__file__).resolve().parents[1] / "lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def row(qid, doc_id, grade, features):
    return {"qid": qid, "keywords": f"query {qid}", "doc_id": doc_id, "grade": grade, "features": features}


class LearningContractTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_line_separators_inside_strings(self):
        records = [{"id": "1", "overview": "first\u0085second\u2028third\u2029last"},
                   {"id": "2", "overview": "line one\nline two"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movies.jsonl"
            path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
            self.assertEqual(records, lab.read_jsonl(path))

    def test_queries_do_not_leak_and_scaler_uses_training_only(self):
        rows = [row(qid, str(i), i % 2, [qid, i, 2000]) for qid in range(20) for i in range(4)]
        training, testing = lab.split_queries(rows)
        self.assertEqual(18, len({r["qid"] for r in training}))
        self.assertEqual(2, len({r["qid"] for r in testing}))
        self.assertFalse({r["qid"] for r in training} & {r["qid"] for r in testing})
        means, stds = lab.fit_scaler(training)
        for record in testing:
            record["features"] = [1e9, 1e9, 1e9]
        means_after, stds_after = lab.fit_scaler(training)
        np.testing.assert_array_equal(means, means_after)
        np.testing.assert_array_equal(stds, stds_after)
        self.assertEqual(1, stds[2])
        self.assertEqual(2000, means[2])

    def test_pairing_never_crosses_queries_and_emits_both_directions(self):
        rows = [row(1, "a", 1, [4, 2, 2000]), row(1, "b", 0, [2, 2, 1990]),
                row(1, "c", 1, [6, 2, 2010]), row(2, "d", 0, [99, 9, 2030])]
        x, y = lab.make_pairs(rows, np.array([0, 0, 0]), np.array([2, 1, 10]))
        self.assertEqual((4, 3), x.shape)
        np.testing.assert_array_equal(x[::2], -x[1::2])
        np.testing.assert_array_equal(y[::2], -y[1::2])
        np.testing.assert_array_equal(x[0], [1, 0, 1])
        self.assertEqual(1, y[0])

    def test_pairing_without_preferences_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no unequal-grade pairs"):
            lab.make_pairs([row(1, "a", 1, [1, 2, 3])], np.zeros(3), np.ones(3))

    def test_precision_uses_fixed_k_and_macro_query_average(self):
        rows = [row(1, "a", 1, [1, 0, 0]), row(2, "b", 0, [2, 0, 0]), row(2, "c", 1, [1, 0, 0])]
        metrics = lab.evaluate(rows, np.array([-1, 0, 0]), np.zeros(3), np.ones(3))
        self.assertAlmostEqual(0.2, metrics["ltr_p_at_5"])
        self.assertEqual(["c", "b"], metrics["per_query"][1]["ltr_top_ids"])

    def test_model_identity_and_standard_normalization_are_stable(self):
        weights, means, stds = np.array([2, -3, 0.5]), np.array([3, 4, 2000]), np.array([2, 4, 10])
        payload = lab.model_payload(weights, means, stds)
        self.assertEqual(payload, lab.model_payload(weights, means, stds))
        self.assertNotEqual(payload["model"]["name"], lab.model_payload(weights + 0.1, means, stds)["model"]["name"])
        self.assertAlmostEqual(-0.5, lab.predict([5, 8, 2010], payload))

    def test_end_to_end_training_artifacts_hold_out_queries(self):
        rows = [row(qid, str(i), i % 2, [i % 2 * 5 + qid / 100, i, 2000 + i])
                for qid in range(20) for i in range(4)]
        with tempfile.TemporaryDirectory() as directory, patch.object(lab, "ARTIFACTS", Path(directory)), patch("sys.stdout", new=io.StringIO()):
            (Path(directory) / "logged_features.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            lab.train_model()
            payload = lab.read_json(Path(directory) / "model.json")
            summary = lab.read_json(Path(directory) / "training.json")
            self.assertEqual(144, summary["training_pairs"])
            self.assertEqual(72, summary["train_rows"])
            self.assertEqual(8, summary["test_rows"])
            self.assertEqual(payload["model"]["name"], summary["model_name"])
            self.assertFalse(set(summary["train_qids"]) & set(summary["test_qids"]))
            self.assertGreater(summary["weights"][0], 0)


class OpenSearchContractTests(unittest.TestCase):
    def test_named_features_are_mapped_in_declared_order_and_keep_zero(self):
        hit = {"_id": "a", "fields": {"_ltrlog": [{"features": [
            {"name": "release_year", "value": 1999}, {"name": "overview_bm25", "value": 0},
            {"name": "title_bm25", "value": 3.5},
        ]}]}}
        self.assertEqual([3.5, 0, 1999], lab.extract_features(hit))
        hit["fields"]["_ltrlog"][0]["features"].pop()
        with self.assertRaisesRegex(ValueError, "expected"):
            lab.extract_features(hit)

    def test_logging_retrieves_nonmatches_and_rejects_missing_documents(self):
        request = lab.log_request("harry potter", ["a", "b"])
        filters = request["query"]["bool"]["filter"]
        self.assertEqual(["a", "b"], filters[0]["ids"]["values"])
        self.assertIn("sltr", filters[1])
        self.assertTrue(request["ext"]["ltr_log"]["log_specs"]["missing_as_zero"])
        with patch.object(lab, "request", return_value={"hits": {"hits": []}}):
            with self.assertRaisesRegex(ValueError, "Missing indexed documents"):
                lab.fetch_features("harry potter", ["a"])

    def test_rescore_replaces_bm25_and_validates_window(self):
        body = lab.search_request("harry potter", 5, 500, "model")
        self.assertEqual("most_fields", body["query"]["multi_match"]["type"])
        self.assertEqual(0, body["rescore"]["query"]["query_weight"])
        self.assertEqual(1, body["rescore"]["query"]["rescore_query_weight"])
        for window in (0, 4, 10001):
            with self.assertRaisesRegex(ValueError, "window"):
                lab.search_request("harry potter", 5, window, "model")

    def test_http_only_suppresses_explicit_404(self):
        for status in (404, 403, 500):
            error = HTTPError("http://test/", status, "error", {}, io.BytesIO(b'{"error":"test"}'))
            with patch.object(lab, "urlopen", side_effect=error):
                if status == 404:
                    self.assertIsNone(lab.request("GET", "/test", missing_ok=True))
                else:
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                        lab.request("GET", "/test", missing_ok=True)

    def test_feature_setup_never_deletes_and_reuses_identical_set(self):
        stored = {"_source": {"featureset": {"features": lab.feature_definitions()}}}
        with patch.object(lab, "request", side_effect=[{}, stored]) as mocked, patch.object(lab, "save_json") as saved, patch("sys.stdout", new=io.StringIO()):
            lab.ensure_features()
            self.assertEqual(["GET", "GET"], [call.args[0] for call in mocked.call_args_list])
            self.assertEqual("featureset_request.json", saved.call_args.args[0])
            self.assertEqual(lab.feature_definitions(), saved.call_args.args[1]["featureset"]["features"])
        stored["_source"]["featureset"]["features"][0]["template"] = {"match_all": {}}
        with patch.object(lab, "request", side_effect=[{}, stored]), patch.object(lab, "save_json"):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                lab.ensure_features()

    def test_logging_exports_social_network_request_or_first_query(self):
        rows = [row(1, "a", 0, [0, 0, 2000]), row(2, "b", 1, [1, 2, 2010])]
        rows[1]["keywords"] = "social network"
        for expected_query, expected_ids in (("social network", ["b"]), ("query 1", ["a"])):
            with tempfile.TemporaryDirectory() as directory, patch.object(lab, "ARTIFACTS", Path(directory)), patch.object(lab, "read_json", return_value=rows), patch.object(lab, "fetch_features", return_value={"a": [0, 0, 2000], "b": [1, 2, 2010]}), patch("sys.stdout", new=io.StringIO()):
                lab.log_judgments()
                payload = json.loads((Path(directory) / "log_request.json").read_text())
                self.assertEqual(lab.log_request(expected_query, expected_ids), payload)
            rows[1]["keywords"] = "query 2"

    def test_model_reuse_accepts_serialized_definition_without_writes(self):
        payload = lab.model_payload(np.ones(3), np.zeros(3), np.ones(3))
        definition = dict(payload["model"]["model"])
        definition["definition"] = json.dumps(definition["definition"])
        response = {"_source": {"model": {
            "model": definition, "feature_set": {"features": lab.feature_definitions()},
        }}}
        with patch.object(lab, "read_json", return_value=payload), patch.object(lab, "request", return_value=response) as mocked, patch("sys.stdout", new=io.StringIO()):
            lab.upload_model()
            self.assertEqual(1, mocked.call_count)
            self.assertEqual("GET", mocked.call_args.args[0])

    def test_model_reuse_matches_observed_float32_normalizer_storage(self):
        # Actual OpenSearch 3.3.2 GET response for ch10_movie_svm_966dfda0c51d.
        stored_model = {
            "type": "model/linear",
            "definition": {"title_bm25": 0.38504126084985474,
                           "overview_bm25": 0.28044707507716304, "release_year": 0.9776399894644355},
            "feature_normalizers": {
                "title_bm25": {"standard": {"mean": 0.7339204, "standard_deviation": 1.6995629}},
                "overview_bm25": {"standard": {"mean": 0.6458834, "standard_deviation": 1.4829316}},
                "release_year": {"standard": {"mean": 1985.8939, "standard_deviation": 123.573524}},
            },
        }
        payload = lab.model_payload(
            np.array(list(stored_model["definition"].values())),
            np.array([0.7339204004131752, 0.6458834090777248, 1985.8939612828663]),
            np.array([1.699562865851218, 1.482931670233471, 123.57352737054518]),
        )
        response = {"_source": {"model": {
            "model": stored_model, "feature_set": {"features": lab.feature_definitions()},
        }}}
        with patch.object(lab, "read_json", return_value=payload), patch.object(lab, "request", return_value=response) as mocked, patch("sys.stdout", new=io.StringIO()):
            lab.upload_model()
            self.assertEqual(1, mocked.call_count)
            self.assertEqual("GET", mocked.call_args.args[0])
            for changed in ("normalizer", "weight", "type", "features"):
                with self.subTest(changed=changed):
                    modified = json.loads(json.dumps(response))
                    model = modified["_source"]["model"]["model"]
                    if changed == "normalizer":
                        params = model["feature_normalizers"]["title_bm25"]["standard"]
                        params["mean"] = float(np.nextafter(np.float32(params["mean"]), np.float32(np.inf)))
                    elif changed == "weight":
                        model["definition"]["title_bm25"] += 1e-12
                    elif changed == "type":
                        model["type"] = "model/ranklib"
                    else:
                        modified["_source"]["model"]["feature_set"]["features"][0]["template"] = {"match_all": {}}
                    mocked.return_value = modified
                    with self.assertRaisesRegex(RuntimeError, "differs"):
                        lab.upload_model()

    def test_search_preserves_negative_scores_and_detects_serving_drift(self):
        payload = lab.model_payload(np.array([1, 0, 0]), np.array([3, 0, 0]), np.ones(3))
        hit = {"_id": "a", "_score": -2.0, "_source": {"title": "A", "release_year": 2000}}
        response = {"hits": {"hits": [hit], "total": {"value": 1, "relation": "eq"}}}
        with patch.object(lab, "read_json", return_value=payload), patch.object(lab, "request", return_value=response), patch.object(lab, "fetch_features", return_value={"a": [1, 0, 2000]}), patch.object(lab, "save_json") as saved, patch("sys.stdout", new=io.StringIO()):
            lab.search("movie")
            result = saved.call_args.args[1]
            self.assertEqual(-2, result["ltr"][0]["python_score"])
            self.assertEqual("passed", result["score_parity"])
            requests = {call.args[0]: call.args[1] for call in saved.call_args_list}
            self.assertEqual(result["requests"]["baseline"], requests["search_bm25_request.json"])
            self.assertEqual(result["requests"]["ltr"], requests["search_ltr_request.json"])
            self.assertEqual(payload["model"]["name"], requests["search_ltr_request.json"]["rescore"]["query"]["rescore_query"]["sltr"]["model"])
            hit["_score"] = 0.0
            with self.assertRaisesRegex(RuntimeError, "score mismatch"):
                lab.search("movie")

    def test_judgments_reject_conflicting_query_text_and_duplicates(self):
        rows = [row(1, "a", 1, [1, 2, 3]), row(1, "b", 0, [2, 3, 4])]
        rows[1]["keywords"] = "different query"
        with self.assertRaisesRegex(ValueError, "inconsistent keywords"):
            lab.validate_rows(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate judgment"):
            lab.validate_rows([rows[0], rows[0]])

    def test_partial_search_results_fail_fast(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete search"):
            lab.checked_hits({"timed_out": True, "hits": {"hits": []}})


if __name__ == "__main__":
    unittest.main()

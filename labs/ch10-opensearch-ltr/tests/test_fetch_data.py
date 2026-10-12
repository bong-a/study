import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("fetch_data", Path(__file__).resolve().parents[1] / "fetch_data.py")
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


class SourceJudgmentTests(unittest.TestCase):
    def test_exact_duplicates_are_recorded_once_without_changing_labels(self):
        rows, removed = fetch_data.parse_judgments(
            "# qid:11: social network*1\n\n1 qid:11 # 37799\n0 qid:11 # 267752\n0 qid:11 # 267752\n"
        )
        self.assertEqual([row["grade"] for row in rows], [1, 0])
        self.assertEqual(rows[0]["keywords"], "social network")
        self.assertEqual(removed, [rows[1]])

    def test_contradictory_labels_are_not_silently_chosen(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            fetch_data.parse_judgments("# qid:1: movie*1\n1 qid:1 # 100\n0 qid:1 # 100\n")

    def test_source_feature_scores_are_ignored_and_recomputed_later(self):
        rows, removed = fetch_data.parse_judgments("# qid:1: star wars*1\n1 qid:1 1:99.1 2:888 # 11 Star Wars\n")
        self.assertNotIn("features", rows[0])
        self.assertEqual(rows[0]["doc_id"], "11")
        self.assertEqual(removed, [])

    def test_unknown_grades_fail_instead_of_being_dropped(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            fetch_data.parse_judgments("# qid:1: movie*1\n3 qid:1 # 100\n")


if __name__ == "__main__":
    unittest.main()

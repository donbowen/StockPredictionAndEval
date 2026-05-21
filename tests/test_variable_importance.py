import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit_app import get_feature_importance_details, load_factors, load_model_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


class VariableImportanceTests(unittest.TestCase):
    def _get_importance_details(self, model_name):
        pipeline = load_model_pipeline(MODELS_DIR / model_name)
        estimator = pipeline.named_steps[list(pipeline.named_steps)[-1]]
        feature_names = pipeline[:-1].get_feature_names_out()
        return get_feature_importance_details(estimator, feature_names)

    def test_hgbr_model_exposes_estimated_importances(self):
        details = self._get_importance_details("hgbr.joblib")

        self.assertIsNotNone(details)
        self.assertIn("Gradient Boosting", details["title"])
        self.assertGreater(len(details["series"]), 0)
        self.assertGreater(details["series"].sum(), 0)

    def test_mlp_model_exposes_estimated_importances(self):
        details = self._get_importance_details("mlp-64-32.joblib")

        self.assertIsNotNone(details)
        self.assertIn("Neural Network", details["title"])
        self.assertGreater(len(details["series"]), 0)
        self.assertGreater(details["series"].sum(), 0)

    def test_load_factors_falls_back_to_local_csv_when_remote_fetch_fails(self):
        load_factors.clear()

        with patch("streamlit_app.pdr.get_data_famafrench", side_effect=RuntimeError("offline")):
            factors = load_factors()

        self.assertGreater(len(factors), 0)
        self.assertIn("mkt_excess", factors.columns)
        self.assertEqual(factors.index.name, "date")
        self.assertEqual(factors.index[0].day, 28)


if __name__ == "__main__":
    unittest.main()

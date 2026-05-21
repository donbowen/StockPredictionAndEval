import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline

from streamlit_app import load_model_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


class LoadModelPipelineTests(unittest.TestCase):
    def test_all_shipped_models_load_with_compatibility_shims(self):
        model_paths = sorted(MODELS_DIR.glob("*.joblib"))

        self.assertTrue(model_paths, "Expected bundled model files for compatibility coverage")

        failures = []
        for model_path in model_paths:
            try:
                pipeline = load_model_pipeline(model_path)
            except Exception as exc:  # pragma: no cover - exercised only on failure
                failures.append(f"{model_path.name}: {type(exc).__name__}: {exc}")
                continue

            self.assertIsInstance(pipeline, Pipeline, model_path.name)
            self.assertTrue(pipeline.named_steps, model_path.name)

        self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()

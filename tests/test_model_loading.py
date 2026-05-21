import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline

from streamlit_app import load_model_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


class LoadModelPipelineTests(unittest.TestCase):
    def test_all_shipped_models_load_with_compatibility_shims(self):
        model_paths = sorted(MODELS_DIR.glob("*.joblib"))

        self.assertGreater(len(model_paths), 0, "Expected bundled model files for compatibility coverage")

        for model_path in model_paths:
            with self.subTest(model=model_path.name):
                pipeline = load_model_pipeline(model_path)

                self.assertIsInstance(pipeline, Pipeline, model_path.name)
                self.assertGreater(len(pipeline.named_steps), 0, model_path.name)


if __name__ == "__main__":
    unittest.main()

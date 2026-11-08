"""Dashboard support for the v2 results (skips when the local outputs are absent)."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from streamlit_app import (RESULTS_DIR, get_feature_importance_details, load_model_pipeline,
                           _get_torch_ensemble_feature_importances)
from stockpred.nn import MLPEnsemble, MLPSpec, build_net

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / RESULTS_DIR


class TorchEnsembleImportanceTests(unittest.TestCase):
    def test_importance_follows_weights_and_averages_seeds(self):
        spec = MLPSpec("t", (4, 3))
        torch.manual_seed(0)
        states = [{k: v.clone() for k, v in build_net(5, spec).state_dict().items()} for _ in range(2)]
        for st in states:                         # make feature 0 dominant
            st["0.weight"][:, 0] = 5.0
        ens = MLPEnsemble(n_in=5, spec=spec, y_scale=1.0, states=states)
        details = get_feature_importance_details(ens, [f"f{i}" for i in range(5)])
        self.assertIn("Neural Network", details["title"])
        s = details["series"]
        self.assertEqual(len(s), 5)
        self.assertEqual(s.idxmax(), "f0")
        one = _get_torch_ensemble_feature_importances(MLPEnsemble(5, spec, 1.0, states[:1]), list(s.index))
        two = _get_torch_ensemble_feature_importances(MLPEnsemble(5, spec, 1.0, states[1:]), list(s.index))
        np.testing.assert_allclose(s.to_numpy(), (one.to_numpy() + two.to_numpy()) / 2)


@unittest.skipUnless(BUNDLE.exists(), "corrected dashboard bundle not built")
class BundleSchemaTests(unittest.TestCase):
    def test_bundle_has_dashboard_schema_and_dashboard_safe_names(self):
        rw = pd.read_csv(BUNDLE / "returns_wide.csv")
        cols = [c for c in rw.columns if c.startswith("ret_")]
        models = {c.split("_")[1] for c in cols}                     # how the app parses model names
        self.assertGreaterEqual(len(models), 8)
        for m in models:
            self.assertEqual(sum(c.startswith(f"ret_{m}_bin") for c in cols), 6)
            np.testing.assert_allclose(rw[f"ret_{m}_binLS"], rw[f"ret_{m}_bin5"] - rw[f"ret_{m}_bin1"], atol=1e-12)
        self.assertEqual(list(pd.read_csv(BUNDLE / "turnover.csv").columns),
                         ["model", "bin", "date", "% buy", "% sell", "% hold"])
        self.assertEqual(list(pd.read_csv(BUNDLE / "port_stats_tall.csv").columns),
                         ["model", "bin", "date", "ret", "std", "nfirms"])

    def test_saved_corrected_nn_pipeline_loads_and_exposes_importances(self):
        files = sorted((REPO_ROOT / "output_corrected/nn_v1/models").glob("MLP_rank_128x64_*.joblib"))
        if not files:
            self.skipTest("saved NN models not present")
        pipe = load_model_pipeline(files[-1])
        est = pipe.named_steps[list(pipe.named_steps)[-1]]
        details = get_feature_importance_details(est, pipe[:-1].get_feature_names_out())
        self.assertIsNotNone(details["series"])
        self.assertGreater(details["series"].sum(), 0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from streamlit_app import load_factors


class LoadFactorsTests(unittest.TestCase):
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

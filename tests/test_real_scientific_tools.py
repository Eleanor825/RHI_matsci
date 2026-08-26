import unittest

from harness_matsci.real_scientific_tools import (
    RealScientificToolSuite,
    material_feature_vector,
    molecular_feature_vector,
    rdkit_descriptors,
)


class RealScientificToolsTests(unittest.TestCase):
    def test_material_features_are_finite_and_fixed_width(self):
        values = material_feature_vector("Li1 Si1 Pd2", "cubic", 225)
        self.assertEqual(len(values), 21)
        self.assertTrue(all(value == value for value in values))

    def test_rdkit_descriptors_match_known_aspirin_formula_scale(self):
        values = rdkit_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertAlmostEqual(values["MW"], 180.16, places=1)
        self.assertGreater(values["TPSA"], 60.0)
        self.assertLess(values["TPSA"], 70.0)

    def test_rdkit_rejects_invalid_smiles(self):
        with self.assertRaises(ValueError):
            rdkit_descriptors("C1(CC")

    def test_molecular_features_are_finite_and_fixed_width(self):
        values = molecular_feature_vector("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertEqual(len(values), 1030)
        self.assertTrue(all(value == value for value in values))

    def test_observe_rejects_unknown_benchmark(self):
        with self.assertRaises(ValueError):
            RealScientificToolSuite().observe({"benchmark": "unknown"})


if __name__ == "__main__":
    unittest.main()

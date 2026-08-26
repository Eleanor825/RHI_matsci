import unittest

from harness_matsci.external_h14 import balanced_group_folds, weighted_group_partition


class ExternalH14Tests(unittest.TestCase):
    def test_balanced_folds_cover_each_group_once(self):
        sizes = {f"group-{index}": index + 1 for index in range(20)}
        folds = balanced_group_folds(sizes, n_folds=5, seed=7)
        flattened = [group for fold in folds for group in fold]
        self.assertEqual(sorted(flattened), sorted(sizes))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_weighted_partition_is_disjoint_and_nonempty(self):
        sizes = {f"group-{index}": 1 + index % 5 for index in range(30)}
        partition = weighted_group_partition(
            sizes,
            weights={"train": 0.50, "feedback": 0.15, "acceptance": 0.20},
            seed=9,
        )
        flattened = [group for groups in partition.values() for group in groups]
        self.assertTrue(all(partition.values()))
        self.assertEqual(sorted(flattened), sorted(sizes))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()

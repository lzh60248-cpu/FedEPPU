import unittest
import numpy as np
import controlled_quantity_heterogeneity as experiment

class ControlledQuantityTest(unittest.TestCase):
    def test_ratios(self):
        self.assertEqual(experiment.SIZE_RATIOS["balanced"],(1,1,1))
        self.assertEqual(experiment.SIZE_RATIOS["moderate"],(6,3,1))
        self.assertEqual(experiment.SIZE_RATIOS["extreme"],(8,1,1))
    def test_totals_are_allocable_for_every_ratio(self):
        totals=(experiment.TOTAL_LABELED,experiment.TOTAL_UNLABELED_POSITIVE,experiment.TOTAL_UNLABELED_NEGATIVE)
        for ratio in experiment.SIZE_RATIOS.values():
            for total in totals:self.assertEqual(int(experiment.allocate(total,ratio).sum()),total)
    def test_priors_are_constant_by_construction(self):
        for ratio in experiment.SIZE_RATIOS.values():
            up=experiment.allocate(experiment.TOTAL_UNLABELED_POSITIVE,ratio)
            un=experiment.allocate(experiment.TOTAL_UNLABELED_NEGATIVE,ratio)
            priors=up/(up+un);self.assertTrue(np.allclose(priors,priors[0]))

if __name__=="__main__":unittest.main()

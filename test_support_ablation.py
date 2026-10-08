import unittest
import numpy as np
from support_ablation_main import SUPPORT_VARIANTS
from FedEPPU_support_ablation import ComplementMassAggregation

class SupportAblationTest(unittest.TestCase):
    def test_factorial_variants(self):
        self.assertEqual(SUPPORT_VARIANTS["no_support"],("original",False))
        self.assertEqual(SUPPORT_VARIANTS["local_only"],("hard",False))
        self.assertEqual(SUPPORT_VARIANTS["global_only"],("original",True))
        self.assertEqual(SUPPORT_VARIANTS["local_global"],("hard",True))
    def test_disabling_global_support_keeps_base_weights(self):
        base=ComplementMassAggregation.weights(np.array([10.,20.,70.]))
        self.assertTrue(np.allclose(base,base.copy()))
    def test_global_support_changes_only_supported_weighting(self):
        base=np.array([.5,.25,.25]);support=np.array([1.,1.,.2])
        adjusted=ComplementMassAggregation.support_aware_weights(base,support)
        self.assertFalse(np.allclose(base,adjusted));self.assertAlmostEqual(float(adjusted.sum()),1.)

if __name__=="__main__":unittest.main()

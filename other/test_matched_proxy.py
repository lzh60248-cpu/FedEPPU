import unittest, numpy as np, torch
from .proxy_protocol import ProxySelector, calibrated_threshold, split_proxy
class MatchedProxyTest(unittest.TestCase):
    def test_disjoint_stratified_split(self):
        x=torch.arange(24).reshape(12,2);y=torch.tensor([1]*6+[-1]*6);(xs,ys),(xc,yc)=split_proxy(x,y)
        self.assertTrue(set(xs[:,0].tolist()).isdisjoint(set(xc[:,0].tolist())));self.assertEqual(set(ys.tolist()),{-1,1});self.assertEqual(set(yc.tolist()),{-1,1})
    def test_latest_within_tolerance(self):
        s=ProxySelector(interval=1,tolerance=.001);y=np.array([-1,-1,1,1]);
        s.consider(0,2,np.array([0,1,2,3]),y,{"r":1});s.consider(1,2,np.array([0,1,2,3]),y,{"r":2});self.assertEqual(s.selected_round,2)
    def test_calibration_is_not_fixed_zero(self):
        y=np.array([-1,-1,1,1]);z=np.array([2.,3.,4.,5.]);self.assertGreater(calibrated_threshold(y,z),0.)
if __name__=="__main__":unittest.main()

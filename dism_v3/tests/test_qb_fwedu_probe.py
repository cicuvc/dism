"""Small exact references for offline quantile calibration diagnostics."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qb_fwedu_probe import chain_counts, histogram_stats, quantile_prices


class QuantileProbeTests(unittest.TestCase):
    def test_two_codes_matches_exact_margin_sort(self):
        scores=torch.tensor([[9.,1.],[8.,3.],[7.,4.],[6.,5.]],dtype=torch.float64)
        price,fit=quantile_prices(scores,32)
        self.assertEqual((scores-price).argmax(-1).tolist(),[0,0,1,1])
        self.assertEqual(fit['mean_score_regret'],1.)
        self.assertEqual(fit['calibrated']['counts'],[2,2])

    def test_three_codes_dual_matches_exhaustive_optimum(self):
        scores=torch.randn(6,3,generator=torch.Generator().manual_seed(23),dtype=torch.float64)
        price,fit=quantile_prices(scores,64)
        best=max(sum(scores[t,c].item() for t,c in enumerate(a))/6
                 for a in itertools.product(range(3),repeat=6) if all(a.count(c)==2 for c in range(3)))
        self.assertAlmostEqual(fit['best_dual'],best,places=12)
        self.assertEqual(fit['calibrated']['counts'],[2,2,2])
        for old,new in zip(fit['trace'],fit['trace'][1:]):
            self.assertLessEqual(new['dual_per_token'],old['dual_per_token']+1e-12)

    def test_chain_counts_respect_boundaries_and_label_changes(self):
        q=np.array([[1,2,3,4]])
        k=q.copy()
        valid=np.ones(4,dtype=bool)
        segments=np.array([0,0,1,1])
        identity=chain_counts(q,k,k,valid,segments)
        np.testing.assert_array_equal(identity[:,:,0],identity[:,:,2])
        self.assertEqual(identity[0,2,0],0)
        changed=chain_counts(q,k,k+9,valid,segments)
        self.assertEqual(changed[:,:,1:].sum(),0)

    def test_frequency_entropy_is_not_active_count(self):
        stats=histogram_stats([99,1,0])
        self.assertEqual(stats['active'],2)
        self.assertLess(stats['effective'],1.1)
        self.assertEqual(stats['max_frequency'],.99)


if __name__=='__main__':
    unittest.main()

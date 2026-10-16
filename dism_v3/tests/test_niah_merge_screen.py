"""Independent checks for diagnostic read-edge interventions (CPU only)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from niah_merge_screen import interventions, normalized, random_mass_control
from niah_short_chain_screen import interventions as short_interventions


class MergeScreenTests(unittest.TestCase):
    def test_short_chain_control_does_not_require_source_for_minimum(self):
        lengths=np.array([1,2,3,4]);features=np.eye(4)
        source=np.array([1.,0,0,0]);candidate=np.array([0.,1,1,1])
        delta,_=short_interventions(lengths,.7,features,source,candidate,np.ones(4,dtype=bool),4,9)
        baseline=normalized(lengths,.7)[0]
        expected=normalized(lengths,.7,np.array([0.,0,1,1]))[0]
        np.testing.assert_allclose(baseline+delta['min_chain3'],expected,atol=1e-15)
        expected=normalized(lengths,.7,np.array([1.,0,1,1]))[0]
        np.testing.assert_allclose(baseline+delta['short_candidate'],expected,atol=1e-15)

    def test_interventions_against_direct_geometric_sums(self):
        lengths=np.array([0,1,2,3,0,4,1,2])
        tau=.7
        raw=np.array([sum(np.exp(tau*k) for k in range(1,l+1)) for l in lengths])
        features=np.random.default_rng(11).normal(size=(8,4))
        source=np.array([0,0,1,0,0,0,0,0.])
        candidate=np.array([1,1,0,0,0,1,0,0.])
        delta,diag=interventions(lengths,tau,features,source,candidate,np.ones(8,dtype=bool),8,19)
        den=1+raw.sum(); num=raw@features; base=num/den
        np.testing.assert_array_equal(delta['sham'],np.zeros(4))
        for mode,desired in dict(content=(num-(raw*candidate)@features)/den,
                                 denominator=num/(den-raw@candidate),
                                 full=((raw*(1-candidate))@features)/(1+(raw*(1-candidate)).sum()),
                                 half=((raw*(1-.5*candidate))@features)/(1+(raw*(1-.5*candidate)).sum()),
                                 source=((raw*(1-source))@features)/(1+(raw*(1-source)).sum())).items():
            np.testing.assert_allclose(base+delta[mode],desired,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(diag['candidate_mass'],raw@candidate/den)

    def test_full_removal_recovers_underflowed_survivor(self):
        lengths=np.array([1,1000])
        mass,_,_=normalized(lengths,10.)
        self.assertEqual(mass[0],0.)
        keep,fallback,logden=normalized(lengths,10.,np.array([1.,0.]))
        self.assertGreater(keep[0],.999)
        self.assertGreater(fallback,0.)
        self.assertTrue(np.isfinite(logden))
        none,fallback,_=normalized(lengths,10.,np.zeros(2))
        np.testing.assert_array_equal(none,np.zeros(2))
        self.assertEqual(fallback,1.)

    def test_random_control_matches_mass_in_each_distance_bin(self):
        query=128
        mass=np.random.default_rng(5).random(129);mass/=mass.sum()+1
        pos=np.arange(129);pool=pos<100;candidate=pool & ((pos%3)==0)
        f=random_mass_control(mass,candidate,pool,query,79)
        self.assertTrue(np.all((f>=0)&(f<=1)))
        self.assertTrue(np.all(f[~pool]==0))
        bins=np.floor(np.log2(np.maximum(1,query-pos))).astype(int)
        for b in np.unique(bins):
            self.assertAlmostEqual(float((mass*f)[bins==b].sum()),float(mass[candidate&(bins==b)].sum()))


if __name__=='__main__':unittest.main()

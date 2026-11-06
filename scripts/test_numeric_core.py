"""Small input-free checks on the distributed KL and paired-statistics helpers."""
from pathlib import Path
import math
import sys
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/annotated_error_kl_20261001'))
sys.path.insert(0,str(ROOT/'experiments/judge_paired_20260928'))
import kl_core
from paired_tests import exact_mcnemar,holm

class NumericChecks(unittest.TestCase):
    def test_identical_distributions(self):
        result=kl_core.distribution_metrics([[0,1,2]],[[0,1,2]],[0])
        self.assertEqual(result['kl_base_to_warm'],[0.0])

    def test_kl_direction(self):
        p=[.8,.2]; q=[.5,.5]
        result=kl_core.distribution_metrics([[math.log(x) for x in p]],
                                            [[math.log(x) for x in q]],[0])
        self.assertAlmostEqual(result['kl_base_to_warm'][0],sum(a*math.log(a/b) for a,b in zip(p,q)))

    def test_causal_alignment(self):
        for prompt in range(1,5):
            for length in range(1,5):
                full=list(range(prompt+length-1))
                self.assertEqual(full[kl_core.response_logit_slice(len(full),prompt,length)],
                                 list(range(prompt-1,prompt+length-1)))

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertEqual(exact_mcnemar(5,5),1)
        self.assertAlmostEqual(exact_mcnemar(0,5),.0625)

    def test_holm(self):
        actual=holm([.01,.04,.03])
        for a,b in zip(actual,[.03,.06,.06]): self.assertAlmostEqual(a,b)

if __name__=='__main__': unittest.main(verbosity=2)

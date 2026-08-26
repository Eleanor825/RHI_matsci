import unittest
from harness_matsci.runtime_trajectory import ScientificActionTrajectory
class RuntimeTrajectoryTests(unittest.TestCase):
 def test_rejects_route_output(self):
  with self.assertRaises(ValueError):
   ScientificActionTrajectory('t','s',False,'b','r',{}, {}, (), (), {'action_worthiness':.5,'route':'proceed'}, {})
 def test_requires_scalar_output(self):
  with self.assertRaises(ValueError):
   ScientificActionTrajectory('t','s',False,'b','r',{}, {}, (), (), {}, {})
if __name__=='__main__':unittest.main()

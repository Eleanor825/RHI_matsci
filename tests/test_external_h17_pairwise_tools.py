import unittest

from harness_matsci.external_h17_pairwise_tools import (
    ExternalH17PairwiseEvidenceEnvironment,
)
from harness_matsci.schema import ActionRecord


class ExternalH17PairwiseToolTests(unittest.TestCase):
    def test_agent_prior_view_preserves_hidden_outcome(self):
        record = ActionRecord(
            record_id="r",
            benchmark="task",
            split="feedback",
            visible_context="context",
            candidate_action="action",
            action_type="choose_candidate",
            evidence=[],
            features={
                "agent_prior_margin": 0.2,
                "agent_prior_uncertainty": 0.1,
                "agent_prior_confidence": 0.8,
                "agent_prior_confidence_low": 0.6,
                "agent_prior_confidence_high": 0.9,
                "agent_prior_estimated_utility": 0.3,
                "agent_prior_estimated_utility_low": 0.1,
                "agent_prior_estimated_utility_high": 0.5,
            },
            label=1,
            utility=0.4,
            metadata={},
        )
        environment = ExternalH17PairwiseEvidenceEnvironment({}, {}, ())
        view = environment.observe_many([record], source="agent_prior")[0]
        self.assertEqual(view.features["tool_estimated_utility"], 0.3)
        self.assertFalse(view.metadata["hidden_outcome_exposed"])
        self.assertFalse(view.metadata["evidence_provenance"]["target_label_exposed"])


if __name__ == "__main__":
    unittest.main()

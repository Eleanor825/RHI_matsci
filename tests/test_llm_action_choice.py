import json
import tempfile
import unittest

from harness_matsci.llm_action_choice import LLMActionChoiceClient, _choice_prompt
from harness_matsci.schema import ActionRecord


class LLMActionChoiceTests(unittest.TestCase):
    def test_prompt_exposes_pre_action_evidence(self):
        prompt = _choice_prompt(
            ActionRecord(
                record_id="r0",
                benchmark="matbot_lfp_historical_choice",
                split="test",
                visible_context="Candidate A and candidate B",
                candidate_action="Execute A rather than B",
                action_type="choose_candidate",
                evidence=["outcome withheld"],
                features={"cost": 0.5},
                label=1,
            )
        )
        self.assertIn("Visible pre-action evidence", prompt)
        self.assertIn("- outcome withheld", prompt)

    def test_prompt_is_invariant_to_hidden_outcome_mutation(self):
        common = {
            "record_id": "r-hidden",
            "benchmark": "matbot_lfp_historical_choice",
            "split": "test",
            "visible_context": "Candidate A and candidate B",
            "candidate_action": "Execute A rather than B",
            "action_type": "choose_candidate",
            "evidence": ["measurement withheld"],
            "features": {"cost": 0.5},
            "label": 1,
        }
        left = ActionRecord(
            **common,
            metadata={
                "hidden_outcome_for_evaluation_only": {"utility_difference": -0.9}
            },
        )
        right = ActionRecord(
            **common,
            metadata={
                "hidden_outcome_for_evaluation_only": {"utility_difference": 0.9}
            },
        )

        self.assertEqual(_choice_prompt(left), _choice_prompt(right))

    def test_parses_numeric_choice_and_caches_by_replica(self):
        calls = []
        response = {
            "output_text": json.dumps(
                {
                    "choice": "A",
                    "p_a_better": 0.7,
                    "p_success_selected": 0.8,
                    "expected_scientific_utility_selected": 0.6,
                    "assessment_confidence": 0.75,
                    "rationale": "supported",
                }
            )
        }

        def request_fn(request, timeout):
            calls.append((request, timeout))
            return json.dumps(response).encode()

        record = ActionRecord(
            record_id="r1",
            benchmark="matbot_lfp_historical_choice",
            split="test",
            visible_context="Candidate A and candidate B",
            candidate_action="Execute A rather than B",
            action_type="choose_candidate",
            evidence=[],
            features={"cost": 0.5, "reversibility": 0.5},
            label=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            client = LLMActionChoiceClient(
                model="test-model",
                api_key="secret",
                base_url="https://example.test",
                cache_path=f"{directory}/cache.json",
                request_fn=request_fn,
            )
            first = client.predict(record, replica_id=0)
            second = client.predict(record, replica_id=0)
            third = client.predict(record, replica_id=1)
        self.assertEqual(first.choice, "A")
        self.assertEqual(first.p_a_better, 0.7)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)
        schema = json.loads(calls[0][0].data)["text"]["format"]["schema"]
        self.assertNotIn("selected_route", schema["properties"])

    def test_retries_structurally_invalid_response(self):
        responses = [
            {"output_text": json.dumps({"choice": "A"})},
            {
                "output_text": json.dumps(
                    {
                        "choice": "B",
                        "p_a_better": 0.3,
                        "p_success_selected": 0.8,
                        "expected_scientific_utility_selected": 0.7,
                        "assessment_confidence": 0.6,
                        "rationale": "retry succeeded",
                    }
                )
            },
        ]

        def request_fn(request, timeout):
            return json.dumps(responses.pop(0)).encode()

        client = LLMActionChoiceClient(
            model="test-model",
            api_key="secret",
            base_url="https://example.test",
            request_fn=request_fn,
            max_retries=1,
        )
        prediction = client.predict(
            ActionRecord(
                record_id="retry",
                benchmark="matbot_lfp_historical_choice",
                split="test",
                visible_context="A versus B",
                candidate_action="Execute A rather than B",
                action_type="choose_candidate",
                evidence=[],
                features={"cost": 0.5},
                label=0,
            ),
            replica_id=0,
        )
        self.assertEqual(prediction.choice, "B")
        self.assertEqual(responses, [])


if __name__ == "__main__":
    unittest.main()

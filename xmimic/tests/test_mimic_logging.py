import unittest

import torch

from humanoidverse.utils.mimic_logging import (
    apply_termination_mask,
    summarize_contact_error_metrics,
    summarize_reward_groups,
)


class MimicLoggingTests(unittest.TestCase):
    def test_apply_termination_mask_logs_raw_applied_and_first_hit(self):
        log_dict = {}
        reset_buf = torch.zeros(3, dtype=torch.bool)
        time_out_buf = torch.zeros(3, dtype=torch.bool)

        reset_buf, time_out_buf, first_mask = apply_termination_mask(
            log_dict,
            reset_buf,
            time_out_buf,
            "motion_far",
            torch.tensor([True, False, True]),
        )
        self.assertTrue(torch.equal(first_mask, torch.tensor([True, False, True])))
        self.assertTrue(torch.equal(reset_buf, torch.tensor([True, False, True])))
        self.assertTrue(torch.equal(time_out_buf, torch.tensor([False, False, False])))
        self.assertAlmostEqual(log_dict["term_raw_motion_far_frac"].item(), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(log_dict["term_applied_motion_far_frac"].item(), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(log_dict["term_first_motion_far_frac"].item(), 2.0 / 3.0, places=6)

        reset_buf, time_out_buf, first_mask = apply_termination_mask(
            log_dict,
            reset_buf,
            time_out_buf,
            "motion_end",
            torch.tensor([False, True, True]),
            torch.tensor([False, True, True]),
            mark_timeout=True,
        )
        self.assertTrue(torch.equal(first_mask, torch.tensor([False, True, False])))
        self.assertTrue(torch.equal(reset_buf, torch.tensor([True, True, True])))
        self.assertTrue(torch.equal(time_out_buf, torch.tensor([False, True, True])))
        self.assertAlmostEqual(log_dict["term_first_motion_end_frac"].item(), 1.0 / 3.0, places=6)

    def test_summarize_reward_groups_splits_positive_penalty_and_termination(self):
        episode_sums = {
            "tracking": torch.tensor([4.0, 2.0]),
            "smoothness": torch.tensor([-1.0, -3.0]),
            "termination": torch.tensor([-2.0, 0.0]),
        }
        reward_scales = {
            "tracking": 0.5,
            "smoothness": -0.25,
            "termination": -4.0,
        }
        metrics = summarize_reward_groups(
            episode_sums=episode_sums,
            reward_scales=reward_scales,
            episode_lengths=torch.tensor([4.0, 2.0]),
            max_episode_length_s=20.0,
            step_dt=0.5,
        )

        self.assertAlmostEqual(metrics["rew_group_positive"].item(), 0.15, places=6)
        self.assertAlmostEqual(metrics["rew_group_penalty"].item(), -0.10, places=6)
        self.assertAlmostEqual(metrics["rew_group_termination"].item(), -0.05, places=6)
        self.assertAlmostEqual(metrics["rew_step_group_positive"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["rew_step_group_penalty"].item(), -0.875, places=6)
        self.assertAlmostEqual(metrics["rew_step_group_termination"].item(), -0.25, places=6)
        self.assertAlmostEqual(metrics["rew_second_group_positive"].item(), 2.0, places=6)
        self.assertAlmostEqual(metrics["rew_second_group_penalty"].item(), -1.75, places=6)
        self.assertAlmostEqual(metrics["rew_second_group_termination"].item(), -0.5, places=6)

    def test_summarize_contact_error_metrics_splits_forbidden_and_object_mismatch(self):
        metrics = summarize_contact_error_metrics(
            window_count=torch.tensor([4.0, 2.0]),
            total_error_count=torch.tensor([3.0, 1.0]),
            forbidden_contact_error_count=torch.tensor([2.0, 0.0]),
            object_contact_mismatch_error_count=torch.tensor([1.0, 1.0]),
            missed_object_contact_error_count=torch.tensor([1.0, 0.0]),
            unexpected_object_contact_error_count=torch.tensor([0.0, 1.0]),
        )

        self.assertAlmostEqual(metrics["contact_error_rate"].item(), 0.625, places=6)
        self.assertTrue(
            torch.allclose(
                metrics["contact_error_rate_per_env"],
                torch.tensor([0.75, 0.5]),
            )
        )
        self.assertAlmostEqual(metrics["forbidden_contact_error_rate"].item(), 0.25, places=6)
        self.assertTrue(
            torch.allclose(
                metrics["forbidden_contact_error_rate_per_env"],
                torch.tensor([0.5, 0.0]),
            )
        )
        self.assertAlmostEqual(metrics["object_contact_mismatch_error_rate"].item(), 0.375, places=6)
        self.assertTrue(
            torch.allclose(
                metrics["object_contact_mismatch_error_rate_per_env"],
                torch.tensor([0.25, 0.5]),
            )
        )
        self.assertAlmostEqual(metrics["missed_object_contact_error_rate"].item(), 0.125, places=6)
        self.assertTrue(
            torch.allclose(
                metrics["missed_object_contact_error_rate_per_env"],
                torch.tensor([0.25, 0.0]),
            )
        )
        self.assertAlmostEqual(metrics["unexpected_object_contact_error_rate"].item(), 0.25, places=6)
        self.assertTrue(
            torch.allclose(
                metrics["unexpected_object_contact_error_rate_per_env"],
                torch.tensor([0.0, 0.5]),
            )
        )


if __name__ == "__main__":
    unittest.main()

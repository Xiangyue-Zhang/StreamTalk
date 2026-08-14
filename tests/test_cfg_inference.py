import unittest


try:
    import torch
    import torch.nn as nn

    from Scripts.FM.Net import (
        DiffusionDITNetPartsFixedExpressions2PostNormInteraction2 as CFGModel,
    )
except ImportError as error:  # The metadata/CLI suite stays runnable without PyTorch.
    torch = None
    nn = None
    CFGModel = None
    IMPORT_ERROR = str(error)
else:
    IMPORT_ERROR = ""


if CFGModel is not None:

    class FakeCFGModel(CFGModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.calls = []

        def forward(self, x, time_steps, **kwargs):
            self.calls.append((x.clone(), time_steps.clone(), kwargs))
            hubert_signal = kwargs["hubert"].mean(dim=-1, keepdim=True)
            expression_signal = kwargs["expressions"].mean(dim=-1, keepdim=True)
            return x + hubert_signal + expression_signal

else:

    class FakeCFGModel:
        pass


@unittest.skipIf(CFGModel is None, f"optional torch CFG test unavailable: {IMPORT_ERROR}")
class CFGInferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeCFGModel()
        self.x = torch.full((2, 3, 1), 10.0)
        self.time_steps = torch.tensor([0.25, 0.75])
        self.hubert = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]
        )
        self.expressions = torch.tensor(
            [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]]
        )
        self.expressions_uncond = torch.tensor(
            [[[0.5], [1.0], [1.5]], [[2.0], [2.5], [3.0]]]
        )
        self.pid = torch.tensor([2, 10])

    def kwargs(self):
        return {
            "hubert": self.hubert,
            "expressions": self.expressions,
            "expressions_uncond": self.expressions_uncond,
            "pid": self.pid,
            "hand_indices": [0],
        }

    def test_cfg_one_preserves_original_forward_path(self):
        result = self.model.forward_with_cfg(
            self.x, self.time_steps, cfg_scale=1.0, **self.kwargs()
        )
        expected = self.x + self.hubert + self.expressions
        torch.testing.assert_close(result, expected)
        self.assertEqual(len(self.model.calls), 1)
        call_x, call_time, call_kwargs = self.model.calls[0]
        torch.testing.assert_close(call_x, self.x)
        torch.testing.assert_close(call_time, self.time_steps)
        self.assertNotIn("expressions_uncond", call_kwargs)

    def test_cfg_three_batches_conditional_and_unconditional_once(self):
        result = self.model.forward_with_cfg(
            self.x, self.time_steps, cfg_scale=3.0, **self.kwargs()
        )
        unconditional = self.x + self.expressions_uncond
        conditional = self.x + self.hubert + self.expressions
        torch.testing.assert_close(
            result, unconditional + 3.0 * (conditional - unconditional)
        )
        self.assertEqual(len(self.model.calls), 1)
        call_x, call_time, call_kwargs = self.model.calls[0]
        self.assertEqual(call_x.shape[0], 4)
        torch.testing.assert_close(call_x[:2], call_x[2:])
        torch.testing.assert_close(call_time[:2], call_time[2:])
        torch.testing.assert_close(call_kwargs["hubert"][:2], torch.zeros_like(self.hubert))
        torch.testing.assert_close(call_kwargs["hubert"][2:], self.hubert)
        torch.testing.assert_close(call_kwargs["expressions"][:2], self.expressions_uncond)
        torch.testing.assert_close(call_kwargs["expressions"][2:], self.expressions)
        torch.testing.assert_close(call_kwargs["pid"], torch.cat((self.pid, self.pid)))
        self.assertEqual(call_kwargs["hand_indices"], [0])

    def test_cfg_zero_runs_only_the_unconditional_branch(self):
        result = self.model.forward_with_cfg(
            self.x, self.time_steps, cfg_scale=0.0, **self.kwargs()
        )
        torch.testing.assert_close(result, self.x + self.expressions_uncond)
        self.assertEqual(len(self.model.calls), 1)
        call_x, _, call_kwargs = self.model.calls[0]
        self.assertEqual(call_x.shape[0], 2)
        torch.testing.assert_close(call_kwargs["hubert"], torch.zeros_like(self.hubert))
        torch.testing.assert_close(call_kwargs["expressions"], self.expressions_uncond)
        torch.testing.assert_close(call_kwargs["pid"], self.pid)

    def test_cfg_rejects_mismatched_unconditional_expressions(self):
        kwargs = self.kwargs()
        kwargs["expressions_uncond"] = self.expressions_uncond[:, :-1]
        with self.assertRaises(ValueError):
            self.model.forward_with_cfg(
                self.x, self.time_steps, cfg_scale=3.0, **kwargs
            )


if __name__ == "__main__":
    unittest.main()

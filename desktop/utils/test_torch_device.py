import unittest

from desktop.utils.torch_device import resolve_pf_device

DEVS = [(0, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"), (1, "NVIDIA GeForce RTX 5060 Ti")]


class TestResolvePfDevice(unittest.TestCase):
    def test_auto_and_cuda_pick_the_5060(self):
        self.assertEqual(resolve_pf_device("auto", devices=DEVS, match="5060"), "cuda:1")
        self.assertEqual(resolve_pf_device("cuda", devices=DEVS, match="5060"), "cuda:1")

    def test_explicit_passthrough(self):
        self.assertEqual(resolve_pf_device("cpu", devices=DEVS), "cpu")
        self.assertEqual(resolve_pf_device("cuda:0", devices=DEVS), "cuda:0")

    def test_no_match_never_picks_the_vllm_gpu(self):
        self.assertEqual(resolve_pf_device("auto", devices=DEVS[:1], match="5060"), "cpu")
        self.assertEqual(resolve_pf_device("auto", devices=[], match="5060"), "cpu")
        with self.assertRaises(RuntimeError):
            resolve_pf_device("cuda", devices=DEVS[:1], match="5060")


if __name__ == "__main__":
    unittest.main()

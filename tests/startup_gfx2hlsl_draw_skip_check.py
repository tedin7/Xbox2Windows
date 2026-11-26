"""A late unavailable renderer must preserve failures already logged by a shader draw check."""
from contextlib import redirect_stdout
import importlib
import io
import subprocess
import unittest
from unittest.mock import patch


class DrawSkipTest(unittest.TestCase):
    def test_earlier_comparison(self):
        for name in ('dpp_pair', 'dpp_quad_broadcast', 'dpp_quad_swap', 'f32_lg', 'f32_nan',
                     'f32_sdwa', 'sdwa_cvt', 'sgpr_spill', 'vopc_sdwa'):
            module = importlib.import_module(f'startup_gfx2hlsl_{name}_check')
            draw = module.run_pixel_words if name == 'sdwa_cvt' else module.draw
            skipped = subprocess.CompletedProcess([], 77, '', 'adapter unavailable\n')
            for passed, expected in ((True, 77), (False, 1)):
                with self.subTest(fixture=name, earlier_check_passed=passed):
                    output = io.StringIO()
                    with patch.object(module, 'failures', []), redirect_stdout(output):
                        module.check('prior pixel comparison', passed)
                        with patch.object(module.subprocess, 'run', return_value=skipped), \
                                patch('gpu_case_runner.run', return_value=skipped):   # the adopted checks draw through it
                            with self.assertRaises(SystemExit) as exit_status:
                                draw('renderer', 'warp', 'vs', 'ps', 'reference')
                    self.assertEqual(exit_status.exception.code, expected)
                    self.assertIn('adapter unavailable', output.getvalue())
                    self.assertIn('prior pixel comparison', output.getvalue())


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
# Test1 QWE
class TestKF(unittest.TestCase):
    # Test that the KF can be constructed with initial position (x) and velocity (v)
    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.3
    
        kf = KF(x, v, 1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    # Test that after calling predict, the mean and covariance have the correct shapes
    def test_after_calling_predict_mean_and_cov_are_of_right_shape(self):
        x = 0.2
        v = 2.3

        kf = KF(x, v, 1.2)
        kf.predict(0.1)

        # Covariance should be a 2x2 matrix, mean should be a vector of length 2
        self.assertEqual(kf.cov.shape, (2, 2))
        self.assertEqual(kf.mean.shape, (2,))

    # Test that calling predict increases the state uncertainty (covariance determinant increases)
    def test_calling_predict_increases_state_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(x, v, 1.2)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(0.1)
            det_after = np.linalg.det(kf.cov)

            # After prediction, uncertainty should increase
            self.assertGreater(det_after, det_before)
            # print(det_before, det_after)  # Uncomment for debugging

    # Test that calling update decreases the state uncertainty (covariance determinant decreases)
    def test_calling_update_decreases_state_uncertainty(self):
        x = 0.2
        v = 2.3
        
        kf = KF(x, v, 1.2)

        det_before = np.linalg.det(kf.cov)
        kf.update(0.1, 0.01)
        det_after = np.linalg.det(kf.cov)

        # After update, uncertainty should decrease
        self.assertLess(det_after, det_before)


if __name__ == "__main__":
    unittest.main()

"""Run directly with Python or discover with unittest; requires TensorFlow."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Layer import VolterraLayer2


class VolterraLayer2Test(unittest.TestCase):
    def test_matches_explicit_causal_polynomial(self):
        rng = np.random.default_rng(42)
        for taps, quadratic_taps in ((3, 2), (2, 4), (1, 1)):
            for length in (1, 7):
                for rank in (2, 3):
                    with self.subTest(taps=taps, q=quadratic_taps, length=length, rank=rank):
                        x = rng.normal(size=(2, length))
                        h1 = rng.normal(size=taps)
                        h2 = rng.normal(size=quadratic_taps * (quadratic_taps + 1) // 2)
                        bias = 0.3
                        layer = VolterraLayer2(taps, quadratic_taps, use_bias=True, dtype="float64")
                        inputs = x if rank == 2 else x[..., None]
                        layer(inputs)
                        layer.set_weights([h1, h2, np.array(bias)])
                        expected = np.full_like(x, bias)
                        for batch in range(len(x)):
                            for n in range(length):
                                for i in range(taps):
                                    if n >= i:
                                        expected[batch, n] += h1[i] * x[batch, n-i]
                                index = 0
                                for i in range(quadratic_taps):
                                    for j in range(i, quadratic_taps):
                                        if n >= j:
                                            expected[batch, n] += h2[index] * x[batch, n-i] * x[batch, n-j]
                                        index += 1
                        actual = layer(inputs).numpy()
                        self.assertEqual(actual.shape, inputs.shape)
                        if rank == 3:
                            actual = actual[..., 0]
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_identity_and_dynamic_graph(self):
        layer = VolterraLayer2(taps=9, quadratic_taps=5)

        @tf.function(input_signature=[tf.TensorSpec([None, None, 1], tf.float32)])
        def apply(x):
            return layer(x)

        for length in (1, 13):
            x = tf.random.normal((2, length, 1), seed=1)
            np.testing.assert_array_equal(apply(x).numpy(), x.numpy())
        self.assertEqual(layer.count_params(), 9 + 5 * 6 // 2)

    def test_gradients_and_optimizer(self):
        layer = VolterraLayer2(2, use_bias=True, dtype="float64")
        x = tf.constant([[0.2, -0.7, 1.1]], dtype=tf.float64)
        layer(x)
        layer.quadratic_kernel.assign([0.3, -0.4, 0.8])
        theoretical, numerical = tf.test.compute_gradient(lambda z: layer(z), [x])
        np.testing.assert_allclose(theoretical[0], numerical[0], atol=1e-9)
        with tf.GradientTape() as tape:
            output_sum = tf.reduce_sum(layer(x))
        gradients = tape.gradient(output_sum, layer.trainable_variables)
        # d(sum y)/d(h1), d(sum y)/d(h2), d(sum y)/d(bias).
        for actual, expected in zip(gradients, ([0.6, -0.5], [1.74, -0.91, 0.53], 3.0)):
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
        before = layer(x).numpy()
        tf.keras.optimizers.SGD(learning_rate=0.01).apply_gradients(
            zip(gradients, layer.trainable_variables)
        )
        self.assertFalse(np.allclose(layer(x).numpy(), before))

    def test_model_save_reload(self):
        inputs = tf.keras.Input(shape=(None, 1))
        layer = VolterraLayer2(3, 2, use_bias=True)
        model = tf.keras.Model(inputs, layer(inputs))
        layer.set_weights([
            np.array([0.8, -0.2, 0.1]), np.array([0.2, 0.4, -0.1]), np.array(0.25),
        ])
        x = tf.constant([[[1.0], [2.0], [-1.0]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volterra.keras"
            model.save(str(path))
            restored = tf.keras.models.load_model(str(path))
            np.testing.assert_allclose(restored(x).numpy(), model(x).numpy())

    def test_invalid_inputs_and_frozen_layer(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                VolterraLayer2(taps=invalid)
            with self.assertRaises(ValueError):
                VolterraLayer2(quadratic_taps=invalid)
        for shape in ((3,), (2, 3, 2), (2, 3, 1, 1)):
            with self.assertRaises(ValueError):
                VolterraLayer2()(tf.zeros(shape))
        layer = VolterraLayer2(trainable=False)
        layer(tf.ones((1, 3)))
        self.assertEqual(layer.trainable_variables, [])


if __name__ == "__main__":
    unittest.main()

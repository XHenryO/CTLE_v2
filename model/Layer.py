import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from sympy import false


class FixedFIRChannelLayer(tf.keras.layers.Layer):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h_init = np.asarray(h, dtype=np.float32).reshape(-1).tolist()

    def build(self, input_shape):
        h = np.asarray(self.h_init, dtype=np.float32)
        h = h / np.sum(h)
        self.h = tf.constant(h.reshape(-1, 1, 1), dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, training=None):
        return tf.nn.conv1d(inputs, self.h, stride=1, padding="VALID")

    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h_init})
        return config


class NormalizeRMSLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        power = tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True)
        return inputs / tf.sqrt(power + tf.keras.backend.epsilon())


class F_Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        Fs=200e9,
        fft_length=None,
        n_bins=None,
        init_real=1.0,
        init_imag=0.0,
        max_abs=1e3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.Fs = Fs
        if fft_length is None:
            if n_bins is None:
                fft_length = 20000
            else:
                fft_length = 2 * (int(n_bins) - 1)
        self.fft_length = int(fft_length)
        self.n_bins = self.fft_length // 2 + 1
        self.init_real = init_real
        self.init_imag = init_imag
        self.max_abs = max_abs

    def build(self, input_shape):
        self.H_real = self.add_weight(
            name="H_real",
            shape=(self.n_bins,),
            initializer=tf.keras.initializers.Constant(self.init_real),
            trainable=True,
        )
        self.H_imag = self.add_weight(
            name="H_imag",
            shape=(self.n_bins,),
            initializer=tf.keras.initializers.Constant(self.init_imag),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("F_Layer expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        n = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            n,
            self.fft_length,
            message="F_Layer input time length must be <= fft_length",
        )

        H_real = self.H_real
        H_imag = self.H_imag
        if self.fft_length % 2 == 0:
            H_imag = tf.concat(
                [tf.zeros([1], dtype=H_imag.dtype), H_imag[1:-1], tf.zeros([1], dtype=H_imag.dtype)],
                axis=0,
            )
        else:
            H_imag = tf.concat([tf.zeros([1], dtype=H_imag.dtype), H_imag[1:]], axis=0)

        H = tf.complex(H_real, H_imag)
        H_abs = tf.abs(H)
        scale = tf.minimum(1.0, tf.cast(self.max_abs, tf.float32) / (H_abs + tf.keras.backend.epsilon()))
        H = H * tf.cast(scale, tf.complex64)
        X = tf.signal.rfft(x, fft_length=[self.fft_length])
        Y = X * H[tf.newaxis, :]
        y = tf.signal.irfft(Y, fft_length=[self.fft_length])
        y = y[:, :n]

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "Fs": self.Fs,
            "fft_length": self.fft_length,
            "n_bins": self.n_bins,
            "init_real": self.init_real,
            "init_imag": self.init_imag,
            "max_abs": self.max_abs,
        })
        return config


class CTLELayer(tf.keras.layers.Layer):
    def __init__(
        self,
        Fs=200e9,
        init_tau_z=1 / (2*np.pi*5e9),
        init_tau_p1=1 / (2*np.pi*25e9),
        init_tau_p2=1 / (2*np.pi*35e9),
        init_A0=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.Fs = Fs
        self.init_tau_z = init_tau_z
        self.init_tau_p1 = init_tau_p1
        self.init_tau_p2 = init_tau_p2
        self.init_A0 = init_A0

    def build(self, input_shape):
        self.log_tau_z = self.add_weight(
            name="log_tau_z",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_z)),
            trainable=True,
        )
        self.log_tau_p1 = self.add_weight(
            name="log_tau_p1",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_p1)),
            trainable=True,
        )
        self.log_tau_p2 = self.add_weight(
            name="log_tau_p2",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_p2)),
            trainable=True,
        )
        self.log_A0 = self.add_weight(
            name="log_A0",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_A0)),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("CTLELayer expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        n = tf.shape(x)[1]

        # 不是2n
        n_fft = n

        freqs = tf.linspace(0.0, self.Fs / 2.0, n_fft // 2 + 1)
        omega = 2.0 * np.pi * freqs
        s = tf.complex(tf.zeros_like(omega), omega)

        tau_z, tau_p1, tau_p2, A0 = self._params()

        H = tf.cast(A0, tf.complex64) * (1.0 + s * tf.cast(tau_z, tf.complex64))
        H = H / (1.0 + s * tf.cast(tau_p1, tf.complex64))
        H = H / (1.0 + s * tf.cast(tau_p2, tf.complex64))

        X = tf.signal.rfft(x, fft_length=[n_fft])
        Y = X * H[tf.newaxis, :]
        y = tf.signal.irfft(Y, fft_length=[n_fft])
        # y = y[:, :n]

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def _params(self):
        tau_z = tf.exp(self.log_tau_z)
        tau_p1 = tf.exp(self.log_tau_p1)
        tau_p2 = tf.exp(self.log_tau_p2)
        A0 = tf.exp(self.log_A0)

        two_pi = tf.constant(2.0 * np.pi, dtype=tau_z.dtype)
        fs = tf.cast(self.Fs, tau_z.dtype)

        fz = 1.0 / (two_pi * tau_z)
        fp1 = 1.0 / (two_pi * tau_p1)
        fp2 = 1.0 / (two_pi * tau_p2)

        fz = tf.clip_by_value(fz, 0.01 * fs, 0.5 * fs)
        fp1 = tf.clip_by_value(fp1, tf.maximum(0.01 * fs, 1.05 * fz), 0.49 * fs)
        fp2 = tf.clip_by_value(fp2, tf.maximum(0.01 * fs, 1.05 * fz), 0.49 * fs)

        tau_z = 1.0 / (two_pi * fz)
        tau_p1 = 1.0 / (two_pi * fp1)
        tau_p2 = 1.0 / (two_pi * fp2)
        A0 = tf.clip_by_value(A0, 1e-6, 1e3)

        return tau_z, tau_p1, tau_p2, A0

    def get_config(self):
        config = super().get_config()
        config.update({
            "Fs": self.Fs,
            "init_tau_z": self.init_tau_z,
            "init_tau_p1": self.init_tau_p1,
            "init_tau_p2": self.init_tau_p2,
            "init_A0": self.init_A0,
        })
        return config


class CTLEBlendLayer(CTLELayer):
    def __init__(self, init_alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha

    def build(self, input_shape):
        super().build(input_shape)
        init_alpha = np.clip(self.init_alpha, 1e-4, 1.0 - 1e-4)
        init_logit_alpha = np.log(init_alpha / (1.0 - init_alpha))
        self.logit_alpha = self.add_weight(
            name="logit_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(init_logit_alpha),
            trainable=True,
        )

    def call(self, inputs, training=None):
        y = super().call(inputs, training=training)
        x = tf.cast(inputs, y.dtype)
        alpha = tf.sigmoid(self.logit_alpha)
        return x + alpha * (y - x)

    def get_config(self):
        config = super().get_config()
        config.update({"init_alpha": self.init_alpha})
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLEBypassGate(tf.keras.layers.Layer):
    """Continuously blend a filtered waveform with an exact bypass path.

    ``output = reference + alpha * (filtered - reference)``.  Initializing
    alpha close to zero lets a larger equalizer topology start from the
    already-validated smaller topology instead of forcing extra equalization.
    """

    def __init__(self, init_alpha=1e-6, filtered_polarity=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(init_alpha)
        self.filtered_polarity = float(filtered_polarity)
        if not 0.0 <= self.init_alpha < 1.0:
            raise ValueError("init_alpha must lie in [0, 1)")
        if not np.isfinite(self.filtered_polarity) or self.filtered_polarity == 0.0:
            raise ValueError("filtered_polarity must be finite and nonzero")

    def build(self, input_shape):
        # A sigmoid gives an interpretable, bounded physical-path enable.
        safe_alpha = np.clip(self.init_alpha, 1e-8, 1.0 - 1e-8)
        initial_logit = float(np.log(safe_alpha / (1.0 - safe_alpha)))
        self.logit_alpha = self.add_weight(
            name="logit_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_logit),
            trainable=True,
        )
        super().build(input_shape)

    def alpha(self):
        return tf.sigmoid(self.logit_alpha)

    def call(self, inputs):
        reference, filtered = inputs
        reference = tf.convert_to_tensor(reference)
        filtered = tf.cast(filtered, reference.dtype)
        return reference + self.alpha() * (filtered - reference)

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_alpha": self.init_alpha,
            "filtered_polarity": self.filtered_polarity,
        })
        return config


class CTLELayer2(CTLELayer):
    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("CTLELayer2 expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        tau_z, tau_p1, tau_p2, A0 = self._params()

        Fs = tf.constant(self.Fs, dtype=x.dtype)
        K = 2.0 * Fs

        b0 = A0 * (tau_z * K + 1.0)
        b1 = A0 * 2.0
        b2 = A0 * (1.0 - tau_z * K)

        d2 = tau_p1 * tau_p2
        d1 = tau_p1 + tau_p2

        a0 = d2 * K * K + d1 * K + 1.0
        a1 = -2.0 * d2 * K * K + 2.0
        a2 = d2 * K * K - d1 * K + 1.0

        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        y = self._iir2(x, b0, b1, b2, a1, a2)

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def _iir2(self, x, b0, b1, b2, a1, a2):
        x_t = tf.transpose(x, [1, 0])

        x0 = tf.zeros_like(x_t[0])
        x1 = tf.zeros_like(x_t[0])
        y0 = tf.zeros_like(x_t[0])
        y1 = tf.zeros_like(x_t[0])

        def step(states, xn):
            xm1, xm2, ym1, ym2 = states
            yn = b0 * xn + b1 * xm1 + b2 * xm2 - a1 * ym1 - a2 * ym2
            return xn, xm1, yn, ym1

        states = tf.scan(
            step,
            x_t,
            initializer=(x0, x1, y0, y1),
        )

        y_t = states[2]
        return tf.transpose(y_t, [1, 0])


class CTLELayer3(CTLELayer):
    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("CTLELayer3 expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        tau_z, tau_p1, tau_p2, A0 = self._params()

        y = self._state_space_filter(x, tau_z, tau_p1, tau_p2, A0)

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def _state_space_filter(self, x, tau_z, tau_p1, tau_p2, A0):
        x_t = tf.transpose(x, [1, 0])
        batch = tf.shape(x)[0]

        q10 = tf.zeros([batch], dtype=x.dtype)
        q20 = tf.zeros([batch], dtype=x.dtype)
        y0 = tf.zeros([batch], dtype=x.dtype)

        dt = 1.0 / tf.cast(self.Fs, x.dtype)
        e1 = tf.exp(-dt / tau_p1)
        e2 = tf.exp(-dt / tau_p2)

        denom = tau_p1 - tau_p2
        eps = tf.constant(1e-18, dtype=x.dtype)
        denom_safe = tf.where(tf.abs(denom) < eps, eps, denom)
        c_general = tau_p1 * (e1 - e2) / denom_safe
        c_limit = e1 * dt / tau_p1
        c = tf.where(tf.abs(denom) < eps, c_limit, c_general)
        zero_gain = tau_z / tau_p2

        def step(carry, xn):
            q1, q2, _ = carry
            q1_next = e1 * q1 + (1.0 - e1) * xn
            q2_next = e2 * q2 + (1.0 - e2) * xn + c * (q1 - xn)
            yn = A0 * (q2_next + zero_gain * (q1_next - q2_next))
            return q1_next, q2_next, yn

        _, _, y_t = tf.scan(
            step,
            x_t,
            initializer=(q10, q20, y0),
        )

        return tf.transpose(y_t, [1, 0])


class CTLELayer4(tf.keras.layers.Layer):
    def __init__(
        self,
        Fs=200e9,
        init_tau_z=1 / (2*np.pi*5e9),
        init_tau_p1=1 / (2*np.pi*25e9),
        init_tau_p2=1 / (2*np.pi*40e9),
        init_A0=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.Fs = Fs
        self.init_tau_z = init_tau_z
        self.init_tau_p1 = init_tau_p1
        self.init_tau_p2 = init_tau_p2
        self.init_A0 = init_A0

    def build(self, input_shape):
        self.log_tau_z = self.add_weight(
            name="log_tau_z",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_z)),
            trainable=True,
        )
        self.log_tau_p1 = self.add_weight(
            name="log_tau_p1",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_p1)),
            trainable=True,
        )
        self.log_tau_p2 = self.add_weight(
            name="log_tau_p2",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_tau_p2)),
            trainable=True,
        )
        self.log_A0 = self.add_weight(
            name="log_A0",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_A0)),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("CTLELayer4 expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        tau_z, tau_p1, tau_p2, A0 = self._prewarped_params()

        Fs = tf.constant(self.Fs, dtype=x.dtype)
        K = 2.0 * Fs

        b0 = A0 * (tau_z * K + 1.0)
        b1 = A0 * 2.0
        b2 = A0 * (1.0 - tau_z * K)

        d2 = tau_p1 * tau_p2
        d1 = tau_p1 + tau_p2

        a0 = d2 * K * K + d1 * K + 1.0
        a1 = -2.0 * d2 * K * K + 2.0
        a2 = d2 * K * K - d1 * K + 1.0

        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        y = self._iir2(x, b0, b1, b2, a1, a2)

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def _target_params(self):
        tau_z = tf.exp(self.log_tau_z)
        tau_p1 = tf.exp(self.log_tau_p1)
        tau_p2 = tf.exp(self.log_tau_p2)
        A0 = tf.exp(self.log_A0)

        two_pi = tf.constant(2.0 * np.pi, dtype=tau_z.dtype)
        fs = tf.cast(self.Fs, tau_z.dtype)

        fz = 1.0 / (two_pi * tau_z)
        fp1 = 1.0 / (two_pi * tau_p1)
        fp2 = 1.0 / (two_pi * tau_p2)

        fz = tf.clip_by_value(fz, 0.01 * fs, 0.25 * fs)
        fp1 = tf.clip_by_value(fp1, tf.maximum(0.05 * fs, 1.05 * fz), 0.499 * fs)
        fp2 = tf.clip_by_value(fp2, tf.maximum(0.05 * fs, 1.005 * fp1), 0.499 * fs)
        A0 = tf.clip_by_value(A0, 1e-6, 1e3)

        return fz, fp1, fp2, A0

    def _prewarped_params(self):
        fz, fp1, fp2, A0 = self._target_params()
        fs = tf.cast(self.Fs, fz.dtype)
        pi = tf.constant(np.pi, dtype=fz.dtype)

        omega_z = 2.0 * fs * tf.tan(pi * fz / fs)
        omega_p1 = 2.0 * fs * tf.tan(pi * fp1 / fs)
        omega_p2 = 2.0 * fs * tf.tan(pi * fp2 / fs)

        tau_z = 1.0 / omega_z
        tau_p1 = 1.0 / omega_p1
        tau_p2 = 1.0 / omega_p2

        return tau_z, tau_p1, tau_p2, A0

    def _iir2(self, x, b0, b1, b2, a1, a2):
        x_t = tf.transpose(x, [1, 0])

        x0 = tf.zeros_like(x_t[0])
        x1 = tf.zeros_like(x_t[0])
        y0 = tf.zeros_like(x_t[0])
        y1 = tf.zeros_like(x_t[0])

        def step(states, xn):
            xm1, xm2, ym1, ym2 = states
            yn = b0 * xn + b1 * xm1 + b2 * xm2 - a1 * ym1 - a2 * ym2
            return xn, xm1, yn, ym1

        states = tf.scan(
            step,
            x_t,
            initializer=(x0, x1, y0, y1),
        )

        y_t = states[2]
        return tf.transpose(y_t, [1, 0])

    def get_config(self):
        config = super().get_config()
        config.update({
            "Fs": self.Fs,
            "init_tau_z": self.init_tau_z,
            "init_tau_p1": self.init_tau_p1,
            "init_tau_p2": self.init_tau_p2,
            "init_A0": self.init_A0,
        })
        return config






class CTLELayer11(tf.keras.layers.Layer):
    def __init__(
        self,
        Fs=100e9,
        init_Rs=50.0,
        init_Cs=10e-12,
        init_RL=50.0,
        init_CL=10e-12,
        init_gm=40e-3,
        Rs_range=(50.0, 1000.0),
        Cs_range=(1e-12, 20e-11),
        RL_range=(50.0, 1000.0),
        CL_range=(1e-12, 20e-11),
        gm_range=(5e-3, 100e-3),
        penalty_weight=100.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.Fs = Fs
        self.init_Rs = init_Rs
        self.init_Cs = init_Cs
        self.init_RL = init_RL
        self.init_CL = init_CL
        self.init_gm = init_gm
        self.Rs_range = tuple(Rs_range)
        self.Cs_range = tuple(Cs_range)
        self.RL_range = tuple(RL_range)
        self.CL_range = tuple(CL_range)
        self.gm_range = tuple(gm_range)
        self.penalty_weight = penalty_weight

    @staticmethod
    def _inv_sigmoid_from_range(value, lo, hi):
        ratio = (value - lo) / (hi - lo)
        ratio = np.clip(ratio, 1e-4, 1.0 - 1e-4)
        return np.log(ratio / (1.0 - ratio))

    @staticmethod
    def _range_map(u, lo, hi):
        return lo + tf.sigmoid(u) * (hi - lo)

    def build(self, input_shape):
        self.u_Rs = self.add_weight(
            name="u_Rs",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_sigmoid_from_range(
                    np.clip(self.init_Rs, self.Rs_range[0], self.Rs_range[1]),
                    self.Rs_range[0],
                    self.Rs_range[1],
                )
            ),
            trainable=True,
        )
        self.u_Cs = self.add_weight(
            name="u_Cs",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_sigmoid_from_range(
                    np.clip(self.init_Cs, self.Cs_range[0], self.Cs_range[1]),
                    self.Cs_range[0],
                    self.Cs_range[1],
                )
            ),
            trainable=True,
        )
        self.u_RL = self.add_weight(
            name="u_RL",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_sigmoid_from_range(
                    np.clip(self.init_RL, self.RL_range[0], self.RL_range[1]),
                    self.RL_range[0],
                    self.RL_range[1],
                )
            ),
            trainable=True,
        )
        self.u_CL = self.add_weight(
            name="u_CL",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_sigmoid_from_range(
                    np.clip(self.init_CL, self.CL_range[0], self.CL_range[1]),
                    self.CL_range[0],
                    self.CL_range[1],
                )
            ),
            trainable=True,
        )
        self.u_gm = self.add_weight(
            name="u_gm",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_sigmoid_from_range(
                    np.clip(self.init_gm, self.gm_range[0], self.gm_range[1]),
                    self.gm_range[0],
                    self.gm_range[1],
                )
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            x = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            x = inputs
        else:
            raise ValueError("CTLELayer11 expects input shape [batch, time, 1] or [batch, time]")

        x = tf.cast(x, tf.float32)
        tau_z, tau_p1, tau_p2, A0 = self._prewarped_params(add_penalty=True)

        Fs = tf.constant(self.Fs, dtype=x.dtype)
        K = 2.0 * Fs

        b0 = A0 * (tau_z * K + 1.0)
        b1 = A0 * 2.0
        b2 = A0 * (1.0 - tau_z * K)

        d2 = tau_p1 * tau_p2
        d1 = tau_p1 + tau_p2

        a0 = d2 * K * K + d1 * K + 1.0
        a1 = -2.0 * d2 * K * K + 2.0
        a2 = d2 * K * K - d1 * K + 1.0

        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        y = self._iir2(x, b0, b1, b2, a1, a2)

        if input_rank == 3:
            return y[:, :, tf.newaxis]
        return y

    def _component_params(self):
        dtype = self.u_Rs.dtype
        Rs = self._range_map(self.u_Rs, tf.cast(self.Rs_range[0], dtype), tf.cast(self.Rs_range[1], dtype))
        Cs = self._range_map(self.u_Cs, tf.cast(self.Cs_range[0], dtype), tf.cast(self.Cs_range[1], dtype))
        RL = self._range_map(self.u_RL, tf.cast(self.RL_range[0], dtype), tf.cast(self.RL_range[1], dtype))
        CL = self._range_map(self.u_CL, tf.cast(self.CL_range[0], dtype), tf.cast(self.CL_range[1], dtype))
        gm = self._range_map(self.u_gm, tf.cast(self.gm_range[0], dtype), tf.cast(self.gm_range[1], dtype))
        return Rs, Cs, RL, CL, gm

    def _target_params(self, add_penalty=False):
        Rs, Cs, RL, CL, gm = self._component_params()
        two_pi = tf.constant(2.0 * np.pi, dtype=Rs.dtype)

        omega_z = 1.0 / (Rs * Cs)
        omega_p1 = (1.0 + gm * Rs / 2.0) / (Rs * Cs)
        omega_p2 = 1.0 / (RL * CL)

        fz = omega_z / two_pi
        fp1 = omega_p1 / two_pi
        fp2 = omega_p2 / two_pi

        gain_scale = gm / CL
        A0 = gain_scale * omega_z / (omega_p1 * omega_p2)
        A0 = tf.clip_by_value(A0, 1e-6, 1e3)

        if add_penalty:
            self._add_frequency_penalty([fz, fp1, fp2])
        return fz, fp1, fp2, A0

    def _add_frequency_penalty(self, freqs):
        if self.penalty_weight == 0:
            return

        fs = tf.cast(self.Fs, freqs[0].dtype)
        f_min = 0.001 * fs
        f_max = 0.499 * fs
        penalty = tf.constant(0.0, dtype=freqs[0].dtype)

        for f in freqs:
            lower = tf.nn.relu((f_min - f) / fs)
            upper = tf.nn.relu((f - f_max) / fs)
            penalty = penalty + lower * lower + upper * upper

        self.add_loss(tf.cast(self.penalty_weight, penalty.dtype) * penalty)

    def _prewarped_params(self, add_penalty=False):
        fz, fp1, fp2, A0 = self._target_params()
        fs = tf.cast(self.Fs, fz.dtype)
        pi = tf.constant(np.pi, dtype=fz.dtype)
        f_min = 0.001 * fs
        f_max = 0.499 * fs

        fz_safe = tf.clip_by_value(fz, f_min, f_max)
        fp1_safe = tf.clip_by_value(fp1, f_min, f_max)
        fp2_safe = tf.clip_by_value(fp2, f_min, f_max)

        omega_z = 2.0 * fs * tf.tan(pi * fz_safe / fs)
        omega_p1 = 2.0 * fs * tf.tan(pi * fp1_safe / fs)
        omega_p2 = 2.0 * fs * tf.tan(pi * fp2_safe / fs)

        tau_z = 1.0 / omega_z
        tau_p1 = 1.0 / omega_p1
        tau_p2 = 1.0 / omega_p2

        return tau_z, tau_p1, tau_p2, A0

    def _iir2(self, x, b0, b1, b2, a1, a2):
        x_t = tf.transpose(x, [1, 0])

        x0 = tf.zeros_like(x_t[0])
        x1 = tf.zeros_like(x_t[0])
        y0 = tf.zeros_like(x_t[0])
        y1 = tf.zeros_like(x_t[0])

        def step(states, xn):
            xm1, xm2, ym1, ym2 = states
            yn = b0 * xn + b1 * xm1 + b2 * xm2 - a1 * ym1 - a2 * ym2
            return xn, xm1, yn, ym1

        states = tf.scan(
            step,
            x_t,
            initializer=(x0, x1, y0, y1),
        )

        y_t = states[2]
        return tf.transpose(y_t, [1, 0])

    def get_config(self):
        config = super().get_config()
        config.update({
            "Fs": self.Fs,
            "init_Rs": self.init_Rs,
            "init_Cs": self.init_Cs,
            "init_RL": self.init_RL,
            "init_CL": self.init_CL,
            "init_gm": self.init_gm,
            "Rs_range": self.Rs_range,
            "Cs_range": self.Cs_range,
            "RL_range": self.RL_range,
            "CL_range": self.CL_range,
            "gm_range": self.gm_range,
            "penalty_weight": self.penalty_weight,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model(tf.keras.layers.Layer):
    """Differentiable TensorFlow implementation of the fitted CTLE v8 model.

    The five trainable variables are physical RS, CS, RL, CL and ibias values.
    Each is represented by an unconstrained scalar and smoothly mapped into its
    calibrated range with a sigmoid.  This class intentionally does not inherit
    from any of the older CTLE layer implementations.
    """

    _FACTOR_NAMES = ("K", "z_low", "z_mid", "z_rhp", "wn", "zeta", "p3", "p4")

    def __init__(
        self,
        Fs=200e9,
        init_RS=1200.0,
        init_CS=120e-15,
        init_RL=1060.0,
        init_CL=0.78e-15,
        init_ibias=300e-6,
        RS_range=(800.0, 1800.0),
        CS_range=(22.4e-15, 200e-15),
        RL_range=(400.0, 1700.0),
        CL_range=(0.1e-15, 1.58e-15),
        ibias_range=(50e-6, 500e-6),
        coefficient_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Fs = float(Fs)
        self.init_RS = float(init_RS)
        self.init_CS = float(init_CS)
        self.init_RL = float(init_RL)
        self.init_CL = float(init_CL)
        self.init_ibias = float(init_ibias)
        self.RS_range = self._validate_range("RS_range", RS_range)
        self.CS_range = self._validate_range("CS_range", CS_range)
        self.RL_range = self._validate_range("RL_range", RL_range)
        self.CL_range = self._validate_range("CL_range", CL_range)
        self.ibias_range = self._validate_range("ibias_range", ibias_range)
        self.coefficient_dir = None if coefficient_dir is None else str(coefficient_dir)

        self._validate_initial("init_RS", self.init_RS, self.RS_range)
        self._validate_initial("init_CS", self.init_CS, self.CS_range)
        self._validate_initial("init_RL", self.init_RL, self.RL_range)
        self._validate_initial("init_CL", self.init_CL, self.CL_range)
        self._validate_initial("init_ibias", self.init_ibias, self.ibias_range)

        base = Path(self.coefficient_dir) if self.coefficient_dir else Path(__file__).resolve().with_name("sim")
        self._coefficient_path = str(base)
        self._v1 = self._read_coefficients(base, "ctle_transfer_ibias_coefficients.json")
        self._v2 = self._read_coefficients(base, "ctle_transfer_ibias_rsrl_coefficients.json")
        self._v3 = self._read_coefficients(base, "ctle_transfer_ibias_rsi_coefficients.json")
        self._v4 = self._read_coefficients(base, "ctle_transfer_ibias_rli_coefficients.json")
        self._v5 = self._read_coefficients(base, "ctle_transfer_ibias_rlcl_coefficients.json")
        self._v6 = self._read_coefficients(base, "ctle_transfer_ibias_cli_coefficients.json")
        self._v7 = self._read_coefficients(base, "ctle_transfer_ibias_rlcli_coefficients.json")
        self._v8 = self._read_coefficients(base, "ctle_transfer_ibias_rscs_coefficients.json")

        self._f0 = float(self._v1["f0_hz"])
        self._w0 = float(self._v1["w0_rad_s"])
        self._ibias0 = float(self._v1["ibias0_A"])
        self._ibias_power = float(self._v1["ibias_power"])
        self._raw_low_np = np.asarray(self._v1["raw_low"], dtype=np.float32)
        self._raw_high_np = np.asarray(self._v1["raw_high"], dtype=np.float32)
        self._m1_np = np.asarray(self._v1["matrix"], dtype=np.float32)
        self._m2_np = np.asarray(self._v2["surface_matrix"], dtype=np.float32)
        self._m3_np = np.asarray(self._v3["surface_matrix"], dtype=np.float32)
        self._m4_np = np.asarray(self._v4["surface_matrix"], dtype=np.float32)
        self._m5_np = np.asarray(self._v5["surface_matrix"], dtype=np.float32)
        self._m6_np = np.asarray(self._v6["surface_matrix"], dtype=np.float32)
        self._m7_np = np.asarray(self._v7["surface_matrix"], dtype=np.float32)
        self._m8_np = np.asarray(self._v8["surface_matrix"], dtype=np.float32)

    @staticmethod
    def _read_coefficients(base, filename):
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(f"CTLE v8 coefficient file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_range(name, values):
        if len(values) != 2:
            raise ValueError(f"{name} must contain (low, high)")
        lo, hi = float(values[0]), float(values[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or lo >= hi:
            raise ValueError(f"invalid {name}: {values}")
        return (lo, hi)

    @staticmethod
    def _validate_initial(name, value, limits):
        if not limits[0] <= value <= limits[1]:
            raise ValueError(f"{name}={value} is outside {limits}")

    @staticmethod
    def _inverse_sigmoid(value, limits):
        lo, hi = limits
        ratio = np.clip((value - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
        return float(np.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _range_map(u, limits):
        dtype = u.dtype
        lo = tf.cast(limits[0], dtype)
        hi = tf.cast(limits[1], dtype)
        return lo + tf.sigmoid(u) * (hi - lo)

    def build(self, input_shape):
        initializer = tf.keras.initializers.Constant
        self.u_RS = self.add_weight(
            name="u_RS", shape=(), initializer=initializer(self._inverse_sigmoid(self.init_RS, self.RS_range)), trainable=True
        )
        self.u_CS = self.add_weight(
            name="u_CS", shape=(), initializer=initializer(self._inverse_sigmoid(self.init_CS, self.CS_range)), trainable=True
        )
        self.u_RL = self.add_weight(
            name="u_RL", shape=(), initializer=initializer(self._inverse_sigmoid(self.init_RL, self.RL_range)), trainable=True
        )
        self.u_CL = self.add_weight(
            name="u_CL", shape=(), initializer=initializer(self._inverse_sigmoid(self.init_CL, self.CL_range)), trainable=True
        )
        self.u_ibias = self.add_weight(
            name="u_ibias", shape=(), initializer=initializer(self._inverse_sigmoid(self.init_ibias, self.ibias_range)), trainable=True
        )
        super().build(input_shape)

    def physical_parameters(self):
        """Return the current differentiable component values in SI units."""
        return {
            "RS": self._range_map(self.u_RS, self.RS_range),
            "CS": self._range_map(self.u_CS, self.CS_range),
            "RL": self._range_map(self.u_RL, self.RL_range),
            "CL": self._range_map(self.u_CL, self.CL_range),
            "ibias": self._range_map(self.u_ibias, self.ibias_range),
        }

    @staticmethod
    def _stack(values):
        return tf.stack([tf.cast(v, tf.float32) for v in values])

    @staticmethod
    def _matmul_features(features, matrix):
        return tf.linalg.matvec(tf.transpose(matrix), features)

    def _q(self, RS, CS, CL, RL, ibias):
        dtype = tf.float32
        raw = self._stack([
            tf.sqrt(tf.cast(RS, dtype) / 1000.0),
            tf.cast(self._w0, dtype) * tf.cast(RS, dtype) * tf.cast(CS, dtype),
            tf.cast(self._w0, dtype) * tf.cast(RL, dtype) * tf.cast(CL, dtype),
            tf.cast(RL, dtype) / 1500.0,
            tf.pow(tf.cast(ibias, dtype) / tf.cast(self._ibias0, dtype), tf.cast(self._ibias_power, dtype)),
        ])
        low = tf.constant(self._raw_low_np, dtype=dtype)
        high = tf.constant(self._raw_high_np, dtype=dtype)
        return 2.0 * (raw - low) / (high - low) - 1.0

    def _base_features(self, RS, CS, CL, RL, ibias):
        q = self._q(RS, CS, CL, RL, ibias)
        return self._stack([
            1.0,
            q[0], q[0]**2, q[0]**3,
            q[1], q[1]**2, q[1]**3,
            q[2], q[2]**2, q[2]**3,
            q[3], q[3]**2, q[3]**3,
            q[4], q[4]**2, q[4]**3, q[4]**4,
        ])

    def _v2_passive_gate(self, CS, CL):
        target = self._v2["surface_slice"]
        anchor = self._v2["zero_correction_anchors"]
        target_cs, target_cl = float(target["CS_F"]), float(target["CL_F"])
        anchor_cs, anchor_cl = float(anchor["CS_F"]), float(anchor["CL_F"])
        td2 = ((CS-target_cs)/(target_cs-anchor_cs))**2 + ((CL-target_cl)/(target_cl-anchor_cl))**2
        ad2 = ((CS-anchor_cs)/(target_cs-anchor_cs))**2 + ((CL-anchor_cl)/(target_cl-anchor_cl))**2
        return tf.exp(-td2) * (1.0-tf.exp(-ad2)) / (1.0-np.exp(-2.0))

    def _log_v2(self, RS, CS, CL, RL, ibias):
        target = self._v2["surface_slice"]
        anchor = self._v2["zero_correction_anchors"]
        tcs,tcl,ti = float(target["CS_F"]),float(target["CL_F"]),float(target["ibias_A"])
        m1=tf.constant(self._m1_np);m2=tf.constant(self._m2_np)
        base_current=self._matmul_features(self._base_features(RS,CS,CL,RL,ibias),m1)
        base_slice=self._matmul_features(self._base_features(RS,tcs,tcl,RL,ti),m1)
        q=self._q(RS,tcs,tcl,RL,ti);sf=self._stack([1.0,q[0],q[0]**2,q[3],q[3]**2,q[0]*q[3]])
        fitted=self._matmul_features(sf,m2)
        qi=self._q(1000.0,CS,CL,1000.0,ibias)[4]
        qi_target=self._q(1000.0,tcs,tcl,1000.0,ti)[4]
        bias_coordinate=(qi-1.0)/(qi_target-1.0)
        return base_current+bias_coordinate*self._v2_passive_gate(CS,CL)*(fitted-base_slice)

    def _v3_passive_gate(self, CS, CL):
        return self._v2_passive_gate(CS, CL)

    def _log_v3(self, RS, CS, CL, RL, ibias):
        target=self._v3["surface_slice"];trl,tcs,tcl=float(target["RL_ohm"]),float(target["CS_F"]),float(target["CL_F"])
        q=self._q(RS,tcs,tcl,trl,ibias);sf=self._stack([1.0,q[0],q[0]**2,q[4],q[4]**2,q[0]*q[4]])
        fitted=self._matmul_features(sf,tf.constant(self._m3_np))
        weight=self._v3_passive_gate(CS,CL)*tf.exp(-((RL-trl)/float(self._v3["rl_decay_scale_ohm"]))**2)
        return self._log_v2(RS,CS,CL,RL,ibias)+weight*(fitted-self._log_v2(RS,tcs,tcl,trl,ibias))

    def _rs_gate(self, RS):
        target_rs=float(self._v4["surface_slice"]["RS_ohm"])
        polynomial=((RS-800.0)*(1600.0-RS)/160000.0)**2
        return polynomial*tf.exp(-((RS-target_rs)/400.0)**4)

    def _log_v4(self, RS, CS, CL, RL, ibias):
        target=self._v4["surface_slice"];trs,tcs,tcl=float(target["RS_ohm"]),float(target["CS_F"]),float(target["CL_F"])
        q=self._q(trs,tcs,tcl,RL,ibias);sf=self._stack([1.0,q[3],q[3]**2,q[4],q[4]**2,q[3]*q[4]])
        fitted=self._matmul_features(sf,tf.constant(self._m4_np))
        return self._log_v3(RS,CS,CL,RL,ibias)+self._v3_passive_gate(CS,CL)*self._rs_gate(RS)*(fitted-self._log_v3(trs,tcs,tcl,RL,ibias))

    def _v5_cs_gate(self, CS):
        target_cs=float(self._v5["surface_slice"]["CS_F"]);g=self._v5["gates"]
        anchor_cs=float(g["CS_anchor_F"]);scale=float(g["CS_scale_F"]);x=(CS-anchor_cs)/(target_cs-anchor_cs)
        return x*tf.exp(-((CS-target_cs)/scale)**4)

    def _log_v5(self, RS, CS, CL, RL, ibias):
        surface=self._v5["surface_slice"];anchor_cl=float(surface["CL_zero_anchor_F"]);target_i=float(surface["ibias_A"])
        r=(RL-1000.0)/600.0;c=(CL-anchor_cl)/0.6e-15
        sf=self._stack([c,c*c,c*r,c*r*r,c*c*r,c*c*r*r])
        i_scale=float(self._v5["gates"]["ibias_scale_A"]);i_gate=tf.exp(-((ibias-target_i)/i_scale)**4)
        correction=self._matmul_features(sf,tf.constant(self._m5_np))
        return self._log_v4(RS,CS,CL,RL,ibias)+self._rs_gate(RS)*self._v5_cs_gate(CS)*i_gate*correction

    def _log_v6(self, RS, CS, CL, RL, ibias):
        surface=self._v6["surface_slice"];scales=self._v6["scales"]
        c=(CL-float(surface["CL_zero_anchor_F"]))/float(scales["CL_F"])
        i=(ibias-float(surface["ibias_zero_anchor_A"]))/float(scales["ibias_A"])
        sf=self._stack([c*i,c*c*i,c*i*i,c*c*i*i])
        rl_gate=tf.exp(-((RL-float(surface["RL_ohm"]))/float(scales["RL_gate_ohm"]))**4)
        correction=self._matmul_features(sf,tf.constant(self._m6_np))
        return self._log_v5(RS,CS,CL,RL,ibias)+self._rs_gate(RS)*self._v5_cs_gate(CS)*rl_gate*correction

    def _log_v7(self, RS, CS, CL, RL, ibias):
        norm=self._v7["normalization"];g=self._v7["gates"];surface=self._v7["surface_slice"]
        r=(RL-float(norm["RL_center_ohm"]))/float(norm["RL_scale_ohm"])
        c=(CL-float(norm["CL_center_F"]))/float(norm["CL_scale_F"])
        i=(ibias-float(norm["ibias_center_A"]))/float(norm["ibias_scale_A"])
        sf=self._stack([1.0,r,c,i,r*r,c*c,i*i])
        anchor_cs=float(g["CS_anchor_F"]);upper=float(g["CS_upper_zero_F"]);target_cs=float(surface["CS_F"])
        raw=((CS-anchor_cs)*(upper-CS))/((target_cs-anchor_cs)*(upper-target_cs))
        cs_gate=raw*raw*tf.exp(-((CS-target_cs)/float(g["CS_scale_F"]))**4)
        correction=self._matmul_features(sf,tf.constant(self._m7_np))
        return self._log_v6(RS,CS,CL,RL,ibias)+self._rs_gate(RS)*cs_gate*correction

    def _log_v8(self, RS, CS, CL, RL, ibias):
        norm=self._v8["normalization"];g=self._v8["gates"]
        r=(RS-float(norm["RS_center_ohm"]))/float(norm["RS_scale_ohm"])
        c=(CS-float(norm["CS_center_F"]))/float(norm["CS_scale_F"])
        sf=self._stack([c*(c+1.0),r*c,r*r*c])
        cs_gate=tf.exp(-tf.pow((CS-float(g["CS_domain_center_F"]))/float(g["CS_domain_scale_F"]),int(g["CS_domain_power"])))
        rl_gate=tf.exp(-((RL-float(g["RL_center_ohm"]))/float(g["RL_scale_ohm"]))**4)
        cl_raw=(CL-float(g["CL_zero_F"]))/(float(g["CL_target_F"])-float(g["CL_zero_F"]))
        cl_gate=cl_raw*tf.exp(-((CL-float(g["CL_target_F"]))/float(g["CL_scale_F"]))**4)
        i_raw=(ibias-float(g["ibias_zero_A"]))/(float(g["ibias_target_A"])-float(g["ibias_zero_A"]))
        i_gate=i_raw*tf.exp(-((ibias-float(g["ibias_target_A"]))/float(g["ibias_scale_A"]))**4)
        correction=self._matmul_features(sf,tf.constant(self._m8_np))
        return self._log_v7(RS,CS,CL,RL,ibias)+self._rs_gate(RS)*cs_gate*rl_gate*cl_gate*i_gate*correction

    def factor_vector(self):
        """Return [K, z_low, z_mid, z_rhp, wn, zeta, p3, p4]."""
        p=self.physical_parameters()
        logs=tf.clip_by_value(self._log_v8(p["RS"],p["CS"],p["CL"],p["RL"],p["ibias"]),-15.0,15.0)
        return tf.exp(logs)

    def frequency_response(self, frequencies_hz):
        """Evaluate the complex v8 transfer function at arbitrary frequencies."""
        f=tf.cast(frequencies_hz,tf.float32);v=self.factor_vector()
        x=tf.complex(tf.zeros_like(f),f/tf.cast(self._f0,f.dtype))
        cv=tf.cast(v,tf.complex64)
        numerator=-cv[0]*(1.0+x/cv[1])*(1.0+x/cv[2])*(1.0-x/cv[3])
        denominator=(1.0+2.0*cv[5]*x/cv[4]+(x/cv[4])**2)*(1.0+x/cv[6])*(1.0+x/cv[7])
        return numerator/denominator

    def call(self, inputs, training=None):
        input_rank=inputs.shape.rank
        if input_rank==3:
            if inputs.shape[-1] not in (1,None):
                raise ValueError("CTLELayer_model expects a singleton channel dimension")
            x=tf.squeeze(inputs,axis=-1)
        elif input_rank==2:
            x=inputs
        else:
            raise ValueError("CTLELayer_model expects input shape [batch, time, 1] or [batch, time]")
        x=tf.cast(x,tf.float32);n=tf.shape(x)[1]
        freqs=tf.linspace(tf.constant(0.0,tf.float32),tf.constant(self.Fs/2.0,tf.float32),n//2+1)
        H=self.frequency_response(freqs)
        X=tf.signal.rfft(x,fft_length=[n]);Y=X*H[tf.newaxis,:]
        y=tf.signal.irfft(Y,fft_length=[n])
        return y[:,:,tf.newaxis] if input_rank==3 else y

    def get_config(self):
        config=super().get_config()
        config.update({
            "Fs":self.Fs,"init_RS":self.init_RS,"init_CS":self.init_CS,"init_RL":self.init_RL,
            "init_CL":self.init_CL,"init_ibias":self.init_ibias,"RS_range":self.RS_range,
            "CS_range":self.CS_range,"RL_range":self.RL_range,"CL_range":self.CL_range,
            "ibias_range":self.ibias_range,"coefficient_dir":self.coefficient_dir,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model2(tf.keras.layers.Layer):
    """Differentiable TensorFlow implementation of the physical CTLE v11 model.

    The layer is intentionally implemented independently from the older CTLE
    layers.  Its five trainable scalars map smoothly to RS, CS, RL, CL and
    ibias while enforcing the conditional physical domain used to fit v11.
    """

    _OP_NAMES = ("gm_S", "gds_S", "cgg_F", "cgs_F", "cgd_F", "cdb_F")
    _CALIBRATION_NAMES = (
        "k_gm",
        "k_gds",
        "k_cgd",
        "k_cgs",
        "k_cdb",
        "k_cds_from_cgg",
        "k_ft_pole",
    )

    def __init__(
        self,
        Fs=200e9,
        init_RS=1200.0,
        init_CS=120e-15,
        init_RL=1060.0,
        init_CL=0.78e-15,
        init_ibias=900e-6,
        RS_range=(200.0, 3000.0),
        ibias_range=(780e-6, 1.35e-3),
        CS_max=600e-15,
        CL_min=0.1e-15,
        source_zero_boundary_hz=20e9,
        coefficient_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Fs = float(Fs)
        self.init_RS = float(init_RS)
        self.init_CS = float(init_CS)
        self.init_RL = float(init_RL)
        self.init_CL = float(init_CL)
        self.init_ibias = float(init_ibias)
        self.RS_range = self._validate_range("RS_range", RS_range)
        self.ibias_range = self._validate_range("ibias_range", ibias_range)
        self.CS_max = float(CS_max)
        self.CL_min = float(CL_min)
        self.source_zero_boundary_hz = float(source_zero_boundary_hz)
        self.coefficient_path = None if coefficient_path is None else str(coefficient_path)

        if self.CS_max <= 0.0 or self.CL_min <= 0.0:
            raise ValueError("CS_max and CL_min must be positive")
        if self.source_zero_boundary_hz <= 0.0:
            raise ValueError("source_zero_boundary_hz must be positive")

        self._validate_initial_domain()
        default_path = (
            Path(__file__).resolve().parent
            / "v11_physics"
            / "results"
            / "ctle_v11_physical_coefficients.json"
        )
        path = Path(self.coefficient_path) if self.coefficient_path else default_path
        if not path.is_file():
            raise FileNotFoundError(f"CTLE v11 coefficient file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_name") != "ctle_v11_physics_informed":
            raise ValueError(f"not a promoted CTLE v11 coefficient file: {path}")
        self._beta_np = np.stack(
            [payload["op_log_polynomial_coefficients"][name] for name in self._OP_NAMES]
        ).astype(np.float32)
        self._calibration_np = np.asarray(
            [payload["calibration"][name] for name in self._CALIBRATION_NAMES],
            dtype=np.float32,
        )

    @staticmethod
    def _validate_range(name, values):
        if len(values) != 2:
            raise ValueError(f"{name} must contain (low, high)")
        low, high = float(values[0]), float(values[1])
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low >= high:
            raise ValueError(f"invalid {name}: {values}")
        return (low, high)

    @staticmethod
    def _inverse_sigmoid(value, limits):
        low, high = limits
        ratio = np.clip((value - low) / (high - low), 1e-6, 1.0 - 1e-6)
        return float(np.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _inverse_log_sigmoid(value, low, high):
        if not low <= value <= high:
            raise ValueError(f"value {value} is outside logarithmic range {(low, high)}")
        ratio = np.clip(
            (np.log(value) - np.log(low)) / (np.log(high) - np.log(low)),
            1e-6,
            1.0 - 1e-6,
        )
        return float(np.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _sigmoid_map(raw, limits):
        dtype = raw.dtype
        low = tf.cast(limits[0], dtype)
        high = tf.cast(limits[1], dtype)
        return low + tf.sigmoid(raw) * (high - low)

    @staticmethod
    def _log_sigmoid_map(raw, low, high):
        dtype = raw.dtype
        low = tf.cast(low, dtype)
        high = tf.cast(high, dtype)
        fraction = tf.sigmoid(raw)
        return low * tf.exp(fraction * (tf.math.log(high) - tf.math.log(low)))

    @staticmethod
    def _rl_bounds_numpy(ibias):
        current_ua = ibias * 1e6
        if current_ua <= 930.0:
            fraction = np.clip((current_ua - 780.0) / 150.0, 0.0, 1.0)
            return 500.0 - 300.0 * fraction, 2000.0 - 400.0 * fraction
        if current_ua < 1000.0:
            fraction = (current_ua - 930.0) / 70.0
            return 200.0, 1600.0 - 400.0 * fraction
        return 200.0, 1200.0

    @staticmethod
    def _cl_upper_numpy(RL):
        if RL <= 1500.0:
            return 32e-15
        return (32.0 - 12.0 * np.clip((RL - 1500.0) / 500.0, 0.0, 1.0)) * 1e-15

    def _cs_lower_numpy(self, RS):
        return 1.0 / (2.0 * np.pi * RS * self.source_zero_boundary_hz)

    def _validate_initial_domain(self):
        if not self.RS_range[0] <= self.init_RS <= self.RS_range[1]:
            raise ValueError(f"init_RS={self.init_RS} is outside {self.RS_range}")
        if not self.ibias_range[0] <= self.init_ibias <= self.ibias_range[1]:
            raise ValueError(f"init_ibias={self.init_ibias} is outside {self.ibias_range}")
        cs_low = self._cs_lower_numpy(self.init_RS)
        if not cs_low <= self.init_CS <= self.CS_max:
            raise ValueError(f"init_CS={self.init_CS} is outside {(cs_low, self.CS_max)}")
        rl_limits = self._rl_bounds_numpy(self.init_ibias)
        if not rl_limits[0] <= self.init_RL <= rl_limits[1]:
            raise ValueError(f"init_RL={self.init_RL} is outside conditional {rl_limits}")
        cl_high = self._cl_upper_numpy(self.init_RL)
        if not self.CL_min <= self.init_CL <= cl_high:
            raise ValueError(f"init_CL={self.init_CL} is outside conditional {(self.CL_min, cl_high)}")

    def build(self, input_shape):
        initializer = tf.keras.initializers.Constant
        rs_raw = self._inverse_sigmoid(self.init_RS, self.RS_range)
        ibias_raw = self._inverse_sigmoid(self.init_ibias, self.ibias_range)
        cs_raw = self._inverse_log_sigmoid(
            self.init_CS, self._cs_lower_numpy(self.init_RS), self.CS_max
        )
        rl_raw = self._inverse_sigmoid(
            self.init_RL, self._rl_bounds_numpy(self.init_ibias)
        )
        cl_raw = self._inverse_log_sigmoid(
            self.init_CL, self.CL_min, self._cl_upper_numpy(self.init_RL)
        )
        self.u_RS = self.add_weight(
            name="u_RS", shape=(), initializer=initializer(rs_raw), trainable=True
        )
        self.u_CS = self.add_weight(
            name="u_CS", shape=(), initializer=initializer(cs_raw), trainable=True
        )
        self.u_RL = self.add_weight(
            name="u_RL", shape=(), initializer=initializer(rl_raw), trainable=True
        )
        self.u_CL = self.add_weight(
            name="u_CL", shape=(), initializer=initializer(cl_raw), trainable=True
        )
        self.u_ibias = self.add_weight(
            name="u_ibias", shape=(), initializer=initializer(ibias_raw), trainable=True
        )
        super().build(input_shape)

    def _conditional_rl_bounds(self, ibias):
        dtype = ibias.dtype
        current_ua = ibias * tf.cast(1e6, dtype)
        first_fraction = tf.clip_by_value((current_ua - 780.0) / 150.0, 0.0, 1.0)
        first_low = 500.0 - 300.0 * first_fraction
        first_high = 2000.0 - 400.0 * first_fraction
        second_fraction = tf.clip_by_value((current_ua - 930.0) / 70.0, 0.0, 1.0)
        second_high = 1600.0 - 400.0 * second_fraction
        low = tf.where(current_ua <= 930.0, first_low, tf.cast(200.0, dtype))
        high = tf.where(
            current_ua <= 930.0,
            first_high,
            tf.where(current_ua < 1000.0, second_high, tf.cast(1200.0, dtype)),
        )
        return low, high

    @staticmethod
    def _conditional_cl_upper(RL):
        dtype = RL.dtype
        fraction = tf.clip_by_value((RL - 1500.0) / 500.0, 0.0, 1.0)
        return tf.cast(32e-15, dtype) - tf.cast(12e-15, dtype) * fraction

    def physical_parameters(self):
        """Return the five differentiable physical parameters in SI units."""
        RS = self._sigmoid_map(self.u_RS, self.RS_range)
        ibias = self._sigmoid_map(self.u_ibias, self.ibias_range)
        rl_low, rl_high = self._conditional_rl_bounds(ibias)
        RL = rl_low + tf.sigmoid(self.u_RL) * (rl_high - rl_low)
        cs_low = 1.0 / (
            tf.cast(2.0 * np.pi * self.source_zero_boundary_hz, RS.dtype) * RS
        )
        CS = self._log_sigmoid_map(self.u_CS, cs_low, self.CS_max)
        cl_high = self._conditional_cl_upper(RL)
        CL = self._log_sigmoid_map(self.u_CL, self.CL_min, cl_high)
        return {"RS": RS, "CS": CS, "RL": RL, "CL": CL, "ibias": ibias}

    def conditional_bounds(self):
        """Return current conditional bounds, useful for training diagnostics."""
        parameters = self.physical_parameters()
        rl_low, rl_high = self._conditional_rl_bounds(parameters["ibias"])
        cs_low = 1.0 / (
            tf.cast(
                2.0 * np.pi * self.source_zero_boundary_hz,
                parameters["RS"].dtype,
            )
            * parameters["RS"]
        )
        return {
            "RS_low": tf.cast(self.RS_range[0], tf.float32),
            "RS_high": tf.cast(self.RS_range[1], tf.float32),
            "CS_low": cs_low,
            "CS_high": tf.cast(self.CS_max, tf.float32),
            "RL_low": rl_low,
            "RL_high": rl_high,
            "CL_low": tf.cast(self.CL_min, tf.float32),
            "CL_high": self._conditional_cl_upper(parameters["RL"]),
            "ibias_low": tf.cast(self.ibias_range[0], tf.float32),
            "ibias_high": tf.cast(self.ibias_range[1], tf.float32),
        }

    def operating_point_parameters(self):
        """Predict v11 MOS small-signal parameters from ibias and RL."""
        parameters = self.physical_parameters()
        ibias = parameters["ibias"]
        RL = parameters["RL"]
        x = tf.math.log(ibias / tf.cast(1e-3, ibias.dtype))
        drain_dc = tf.cast(1.8, ibias.dtype) - ibias * RL
        h = (drain_dc - tf.cast(0.9, ibias.dtype)) / tf.cast(0.6, ibias.dtype)
        features = tf.stack(
            [
                tf.ones_like(x),
                x,
                h,
                x**2,
                x * h,
                h**2,
                x**3,
                x**2 * h,
                x * h**2,
                h**3,
            ]
        )
        log_values = tf.linalg.matvec(tf.constant(self._beta_np), features)
        values = tf.exp(log_values)
        return {name: values[index] for index, name in enumerate(self._OP_NAMES)}

    def frequency_response(self, frequencies_hz):
        """Evaluate the complex v11 H(j*2*pi*f) with TensorFlow operations."""
        frequency = tf.cast(frequencies_hz, tf.float32)
        parameters = self.physical_parameters()
        op = self.operating_point_parameters()
        calibration = tf.constant(self._calibration_np, dtype=tf.float32)
        k_gm, k_gds, k_cgd, k_cgs, k_cdb, k_cds, k_ft = tf.unstack(calibration)

        gm = k_gm * op["gm_S"]
        gds = k_gds * op["gds_S"]
        cgg = op["cgg_F"]
        cgd = k_cgd * op["cgd_F"]
        cgs = k_cgs * op["cgs_F"]
        cdb = k_cdb * op["cdb_F"]
        cds = k_cds * cgg
        s = tf.complex(tf.zeros_like(frequency), tf.cast(2.0 * np.pi, tf.float32) * frequency)

        def complex_value(value):
            return tf.cast(value, tf.complex64)

        mdd = complex_value(1.0 / parameters["RL"] + gds) + s * complex_value(
            parameters["CL"] + cdb + cgd + cds
        )
        mds = -complex_value(gm + gds) - s * complex_value(cds)
        msd = -complex_value(gds) - s * complex_value(cds)
        mss = complex_value(2.0 / parameters["RS"] + gm + gds) + s * complex_value(
            2.0 * parameters["CS"] + cgs + cds
        )
        bd = s * complex_value(cgd) - complex_value(gm)
        bs = complex_value(gm) + s * complex_value(cgs)
        core = (bd * mss - mds * bs) / (mdd * mss - mds * msd)
        ft_hz = gm / (tf.cast(2.0 * np.pi, tf.float32) * cgg)
        transport = tf.cast(1.0, tf.complex64) + s / complex_value(
            tf.cast(2.0 * np.pi, tf.float32) * k_ft * ft_hz
        )
        return core / transport

    def call(self, inputs, training=None):
        input_rank = inputs.shape.rank
        if input_rank == 3:
            if inputs.shape[-1] not in (1, None):
                raise ValueError("CTLELayer_model2 expects a singleton channel dimension")
            waveform = tf.squeeze(inputs, axis=-1)
        elif input_rank == 2:
            waveform = inputs
        else:
            raise ValueError(
                "CTLELayer_model2 expects input shape [batch, time, 1] or [batch, time]"
            )
        waveform = tf.cast(waveform, tf.float32)
        length = tf.shape(waveform)[1]
        frequencies = tf.linspace(
            tf.constant(0.0, tf.float32),
            tf.constant(self.Fs / 2.0, tf.float32),
            length // 2 + 1,
        )
        response = self.frequency_response(frequencies)
        spectrum = tf.signal.rfft(waveform, fft_length=[length])
        output = tf.signal.irfft(
            spectrum * response[tf.newaxis, :], fft_length=[length]
        )
        return output[:, :, tf.newaxis] if input_rank == 3 else output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "Fs": self.Fs,
                "init_RS": self.init_RS,
                "init_CS": self.init_CS,
                "init_RL": self.init_RL,
                "init_CL": self.init_CL,
                "init_ibias": self.init_ibias,
                "RS_range": self.RS_range,
                "ibias_range": self.ibias_range,
                "CS_max": self.CS_max,
                "CL_min": self.CL_min,
                "source_zero_boundary_hz": self.source_zero_boundary_hz,
                "coefficient_path": self.coefficient_path,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model_v12(tf.keras.layers.Layer):
    """Independent differentiable implementation of the physical CTLE v12.

    Five unconstrained trainable scalars map continuously into the expanded,
    conditional RS/CS/RL/CL/ibias domain.  This class deliberately does not
    inherit from CTLELayer_model2 or any other CTLE implementation.
    """

    _OP_NAMES = ("gm_S", "gds_S", "cgg_F", "cgs_F", "cgd_F", "cdb_F")
    _CALIBRATION_NAMES = (
        "k_gm", "k_gds", "k_cgd", "k_cgs", "k_cdb",
        "k_cds_from_cgg", "k_ft_pole",
    )

    def __init__(
        self,
        Fs=200e9,
        init_RS=1200.0,
        init_CS=120e-15,
        init_RL=1000.0,
        init_CL=1e-15,
        init_ibias=1e-3,
        RS_range=(100.0, 4000.0),
        ibias_range=(650e-6, 1.55e-3),
        CS_max=900e-15,
        CL_min=0.05e-15,
        coefficient_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Fs = float(Fs)
        self.init_RS = float(init_RS)
        self.init_CS = float(init_CS)
        self.init_RL = float(init_RL)
        self.init_CL = float(init_CL)
        self.init_ibias = float(init_ibias)
        self.RS_range = self._validate_range("RS_range", RS_range)
        self.ibias_range = self._validate_range("ibias_range", ibias_range)
        self.CS_max = float(CS_max)
        self.CL_min = float(CL_min)
        self.coefficient_path = None if coefficient_path is None else str(coefficient_path)
        if self.CS_max <= 0.0 or self.CL_min <= 0.0:
            raise ValueError("CS_max and CL_min must be positive")

        self._validate_initial_domain()
        default_path = (
            Path(__file__).resolve().parent
            / "v12_physics" / "results" / "ctle_v12_physical_coefficients.json"
        )
        path = Path(self.coefficient_path) if self.coefficient_path else default_path
        if not path.is_file():
            raise FileNotFoundError(f"CTLE v12 coefficient file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_name") != "ctle_v12_physics_informed":
            raise ValueError(f"not a promoted CTLE v12 coefficient file: {path}")
        self._beta_np = np.stack(
            [payload["op_log_polynomial_coefficients"][name] for name in self._OP_NAMES]
        ).astype(np.float32)
        self._feature_powers_np = np.asarray(
            payload["feature_definition"]["feature_powers_x_h"], dtype=np.int32
        )
        self._calibration_np = np.asarray(
            [payload["calibration"][name] for name in self._CALIBRATION_NAMES],
            dtype=np.float32,
        )

    @staticmethod
    def _validate_range(name, values):
        if len(values) != 2:
            raise ValueError(f"{name} must contain (low, high)")
        low, high = float(values[0]), float(values[1])
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low >= high:
            raise ValueError(f"invalid {name}: {values}")
        return low, high

    @staticmethod
    def _inverse_sigmoid(value, limits):
        low, high = limits
        ratio = np.clip((value - low) / (high - low), 1e-6, 1.0 - 1e-6)
        return float(np.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _inverse_log_sigmoid(value, low, high):
        if not low <= value <= high:
            raise ValueError(f"value {value} is outside logarithmic range {(low, high)}")
        ratio = np.clip(
            (np.log(value) - np.log(low)) / (np.log(high) - np.log(low)),
            1e-6, 1.0 - 1e-6,
        )
        return float(np.log(ratio / (1.0 - ratio)))

    @staticmethod
    def _sigmoid_map(raw, limits):
        low = tf.cast(limits[0], raw.dtype)
        high = tf.cast(limits[1], raw.dtype)
        return low + tf.sigmoid(raw) * (high - low)

    @staticmethod
    def _log_sigmoid_map(raw, low, high):
        low = tf.cast(low, raw.dtype)
        high = tf.cast(high, raw.dtype)
        return low * tf.exp(tf.sigmoid(raw) * (tf.math.log(high) - tf.math.log(low)))

    @staticmethod
    def _rl_bounds_numpy(ibias):
        current = ibias * 1e6
        if current <= 740.0:
            return 1400.0, 2200.0
        if current <= 770.0:
            return 1400.0, 2200.0 - 200.0 * (current - 740.0) / 30.0
        if current < 780.0:
            return 1400.0 - 900.0 * (current - 770.0) / 10.0, 2000.0
        if current <= 930.0:
            fraction = (current - 780.0) / 150.0
            return 500.0 - 300.0 * fraction, 2000.0 - 400.0 * fraction
        if current <= 1000.0:
            return 200.0, 1600.0
        if current <= 1350.0:
            fraction = (current - 1000.0) / 350.0
            return 200.0 - 100.0 * fraction, 1600.0 - 400.0 * fraction
        fraction = (current - 1350.0) / 200.0
        return 100.0, 1200.0 - 400.0 * fraction

    @staticmethod
    def _cl_upper_numpy(RL):
        if RL <= 1600.0:
            return 50e-15
        if RL <= 2100.0:
            return (50.0 - 10.0 * (RL - 1600.0) / 500.0) * 1e-15
        return (40.0 - 8.0 * (RL - 2100.0) / 100.0) * 1e-15

    @staticmethod
    def _cs_lower_numpy(RS):
        # Conservative envelope through the verified expanded RS-CS points:
        # 60 fF at 100 ohm and 1.5 fF at 4 kohm.
        return 6000e-15 / RS

    def _validate_initial_domain(self):
        if not self.RS_range[0] <= self.init_RS <= self.RS_range[1]:
            raise ValueError(f"init_RS={self.init_RS} is outside {self.RS_range}")
        if not self.ibias_range[0] <= self.init_ibias <= self.ibias_range[1]:
            raise ValueError(f"init_ibias={self.init_ibias} is outside {self.ibias_range}")
        cs_low = self._cs_lower_numpy(self.init_RS)
        if not cs_low <= self.init_CS <= self.CS_max:
            raise ValueError(f"init_CS={self.init_CS} is outside {(cs_low, self.CS_max)}")
        rl_limits = self._rl_bounds_numpy(self.init_ibias)
        if not rl_limits[0] <= self.init_RL <= rl_limits[1]:
            raise ValueError(f"init_RL={self.init_RL} is outside conditional {rl_limits}")
        cl_high = self._cl_upper_numpy(self.init_RL)
        if not self.CL_min <= self.init_CL <= cl_high:
            raise ValueError(f"init_CL={self.init_CL} is outside conditional {(self.CL_min, cl_high)}")

    def build(self, input_shape):
        initializer = tf.keras.initializers.Constant
        rs_raw = self._inverse_sigmoid(self.init_RS, self.RS_range)
        ibias_raw = self._inverse_sigmoid(self.init_ibias, self.ibias_range)
        cs_raw = self._inverse_log_sigmoid(
            self.init_CS, self._cs_lower_numpy(self.init_RS), self.CS_max
        )
        rl_raw = self._inverse_sigmoid(
            self.init_RL, self._rl_bounds_numpy(self.init_ibias)
        )
        cl_raw = self._inverse_log_sigmoid(
            self.init_CL, self.CL_min, self._cl_upper_numpy(self.init_RL)
        )
        self.u_RS = self.add_weight(name="u_RS", shape=(), initializer=initializer(rs_raw), trainable=True)
        self.u_CS = self.add_weight(name="u_CS", shape=(), initializer=initializer(cs_raw), trainable=True)
        self.u_RL = self.add_weight(name="u_RL", shape=(), initializer=initializer(rl_raw), trainable=True)
        self.u_CL = self.add_weight(name="u_CL", shape=(), initializer=initializer(cl_raw), trainable=True)
        self.u_ibias = self.add_weight(name="u_ibias", shape=(), initializer=initializer(ibias_raw), trainable=True)
        super().build(input_shape)

    def _conditional_rl_bounds(self, ibias):
        dtype = ibias.dtype
        current = ibias * tf.cast(1e6, dtype)
        f_740_770 = tf.clip_by_value((current - 740.0) / 30.0, 0.0, 1.0)
        f_770_780 = tf.clip_by_value((current - 770.0) / 10.0, 0.0, 1.0)
        f_780_930 = tf.clip_by_value((current - 780.0) / 150.0, 0.0, 1.0)
        f_1000_1350 = tf.clip_by_value((current - 1000.0) / 350.0, 0.0, 1.0)
        f_1350_1550 = tf.clip_by_value((current - 1350.0) / 200.0, 0.0, 1.0)
        low = tf.where(
            current <= 740.0, 1400.0,
            tf.where(
                current <= 770.0, 1400.0,
                tf.where(
                    current < 780.0, 1400.0 - 900.0 * f_770_780,
                    tf.where(
                        current <= 930.0, 500.0 - 300.0 * f_780_930,
                        tf.where(
                            current <= 1000.0, 200.0,
                            tf.where(current <= 1350.0, 200.0 - 100.0 * f_1000_1350, 100.0),
                        ),
                    ),
                ),
            ),
        )
        high = tf.where(
            current <= 740.0, 2200.0,
            tf.where(
                current <= 770.0, 2200.0 - 200.0 * f_740_770,
                tf.where(
                    current < 780.0, 2000.0,
                    tf.where(
                        current <= 930.0, 2000.0 - 400.0 * f_780_930,
                        tf.where(
                            current <= 1000.0, 1600.0,
                            tf.where(
                                current <= 1350.0, 1600.0 - 400.0 * f_1000_1350,
                                1200.0 - 400.0 * f_1350_1550,
                            ),
                        ),
                    ),
                ),
            ),
        )
        return tf.cast(low, dtype), tf.cast(high, dtype)

    @staticmethod
    def _conditional_cl_upper(RL):
        dtype = RL.dtype
        first = tf.cast(50e-15, dtype) - tf.cast(10e-15, dtype) * tf.clip_by_value(
            (RL - 1600.0) / 500.0, 0.0, 1.0
        )
        second = tf.cast(40e-15, dtype) - tf.cast(8e-15, dtype) * tf.clip_by_value(
            (RL - 2100.0) / 100.0, 0.0, 1.0
        )
        return tf.where(RL <= 2100.0, first, second)

    def physical_parameters(self):
        RS = self._sigmoid_map(self.u_RS, self.RS_range)
        ibias = self._sigmoid_map(self.u_ibias, self.ibias_range)
        rl_low, rl_high = self._conditional_rl_bounds(ibias)
        RL = rl_low + tf.sigmoid(self.u_RL) * (rl_high - rl_low)
        cs_low = tf.cast(6000e-15, RS.dtype) / RS
        CS = self._log_sigmoid_map(self.u_CS, cs_low, self.CS_max)
        CL = self._log_sigmoid_map(
            self.u_CL, tf.cast(self.CL_min, RL.dtype), self._conditional_cl_upper(RL)
        )
        return {"RS": RS, "CS": CS, "RL": RL, "CL": CL, "ibias": ibias}

    def conditional_bounds(self):
        p = self.physical_parameters()
        rl_low, rl_high = self._conditional_rl_bounds(p["ibias"])
        return {
            "RS_low": tf.cast(self.RS_range[0], tf.float32),
            "RS_high": tf.cast(self.RS_range[1], tf.float32),
            "CS_low": tf.cast(6000e-15, tf.float32) / p["RS"],
            "CS_high": tf.cast(self.CS_max, tf.float32),
            "RL_low": rl_low,
            "RL_high": rl_high,
            "CL_low": tf.cast(self.CL_min, tf.float32),
            "CL_high": self._conditional_cl_upper(p["RL"]),
            "ibias_low": tf.cast(self.ibias_range[0], tf.float32),
            "ibias_high": tf.cast(self.ibias_range[1], tf.float32),
        }


    def operating_point_parameters(self):
        p = self.physical_parameters()
        x = tf.math.log(p["ibias"] / tf.cast(1e-3, tf.float32))
        h = ((tf.cast(1.8, tf.float32) - p["ibias"] * p["RL"]) - 0.9) / 0.6

        # Avoid tf.pow(base, 0) here.  Its backward formula can evaluate
        # 0 * base**-1 and produce NaN exactly at x=0 (ibias=1 mA), even
        # though the mathematical derivative of a zeroth-order term is zero.
        def integer_power(base, exponent):
            return tf.ones_like(base) if int(exponent) == 0 else base ** int(exponent)

        features = tf.stack([
            integer_power(x, xp) * integer_power(h, hp)
            for xp, hp in self._feature_powers_np
        ])
        values = tf.exp(tf.linalg.matvec(tf.constant(self._beta_np), features))
        return {name: values[index] for index, name in enumerate(self._OP_NAMES)}

    def frequency_response(self, frequencies_hz):
        frequency = tf.cast(frequencies_hz, tf.float32)
        p = self.physical_parameters()
        op = self.operating_point_parameters()
        k_gm, k_gds, k_cgd, k_cgs, k_cdb, k_cds, k_ft = tf.unstack(
            tf.constant(self._calibration_np, dtype=tf.float32)
        )
        gm = k_gm * op["gm_S"]
        gds = k_gds * op["gds_S"]
        cgg = op["cgg_F"]
        cgd = k_cgd * op["cgd_F"]
        cgs = k_cgs * op["cgs_F"]
        cdb = k_cdb * op["cdb_F"]
        cds = k_cds * cgg
        s = tf.complex(tf.zeros_like(frequency), tf.cast(2.0 * np.pi, tf.float32) * frequency)

        def cv(value):
            return tf.cast(value, tf.complex64)

        mdd = cv(1.0 / p["RL"] + gds) + s * cv(p["CL"] + cdb + cgd + cds)
        mds = -cv(gm + gds) - s * cv(cds)
        msd = -cv(gds) - s * cv(cds)
        mss = cv(2.0 / p["RS"] + gm + gds) + s * cv(2.0 * p["CS"] + cgs + cds)
        bd = s * cv(cgd) - cv(gm)
        bs = cv(gm) + s * cv(cgs)
        core = (bd * mss - mds * bs) / (mdd * mss - mds * msd)
        ft_hz = gm / (tf.cast(2.0 * np.pi, tf.float32) * cgg)
        return core / (tf.cast(1.0, tf.complex64) + s / cv(tf.cast(2.0 * np.pi, tf.float32) * k_ft * ft_hz))

    def call(self, inputs, training=None):
        rank = inputs.shape.rank
        if rank == 3:
            if inputs.shape[-1] not in (1, None):
                raise ValueError("CTLELayer_model_v12 expects a singleton channel dimension")
            waveform = tf.squeeze(inputs, axis=-1)
        elif rank == 2:
            waveform = inputs
        else:
            raise ValueError("CTLELayer_model_v12 expects [batch,time] or [batch,time,1]")
        waveform = tf.cast(waveform, tf.float32)
        length = tf.shape(waveform)[1]
        frequencies = tf.linspace(0.0, tf.cast(self.Fs / 2.0, tf.float32), length // 2 + 1)
        spectrum = tf.signal.rfft(waveform, fft_length=[length])
        output = tf.signal.irfft(
            spectrum * self.frequency_response(frequencies)[tf.newaxis, :],
            fft_length=[length],
        )
        return output[:, :, tf.newaxis] if rank == 3 else output

    def get_config(self):
        config = super().get_config()
        config.update({
            "Fs": self.Fs,
            "init_RS": self.init_RS,
            "init_CS": self.init_CS,
            "init_RL": self.init_RL,
            "init_CL": self.init_CL,
            "init_ibias": self.init_ibias,
            "RS_range": self.RS_range,
            "ibias_range": self.ibias_range,
            "CS_max": self.CS_max,
            "CL_min": self.CL_min,
            "coefficient_path": self.coefficient_path,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model_v12_v2(CTLELayer_model_v12):
    """v12 physical CTLE with circular context and true FFT linear convolution.

    The inherited v12 code still owns the five trainable raw variables, their
    conditional physical mappings, the MOS operating-point surface and the
    complex continuous-time response.  This subclass changes only the signal
    filtering path:

      circular context -> delayed finite causal FIR -> zero-padded FFT linear
      convolution -> delay-compensated crop of the original block.

    ``guard_time_s`` is the physical memory budget.  With the Model_ch_Post
    sampling rate of 100 GSa/s its default 10.24 ns gives 1024 context samples.
    Scaling the sample rate therefore scales the number of samples while
    preserving the same physical time window.
    """

    def __init__(
        self,
        Fs=100e9,
        guard_time_s=10.24e-9,
        guard_samples=None,
        impulse_length=None,
        impulse_fft_oversampling=8,
        **kwargs,
    ):
        super().__init__(Fs=Fs, **kwargs)
        self.guard_time_s = float(guard_time_s)
        if not np.isfinite(self.guard_time_s) or self.guard_time_s <= 0.0:
            raise ValueError("guard_time_s must be positive and finite")

        automatic_guard = int(np.ceil(self.Fs * self.guard_time_s))
        self.guard_samples = (
            automatic_guard if guard_samples is None else int(guard_samples)
        )
        self.impulse_length = (
            self.guard_samples if impulse_length is None else int(impulse_length)
        )
        self.impulse_fft_oversampling = int(impulse_fft_oversampling)
        if self.guard_samples < 1:
            raise ValueError("guard_samples must be at least 1")
        if self.impulse_length < 2:
            raise ValueError("impulse_length must be at least 2")
        if self.impulse_fft_oversampling < 2:
            raise ValueError("impulse_fft_oversampling must be at least 2")

        # Half the FIR length is a purely numerical delay.  It moves the
        # band-limited impulse away from the IFFT boundary so that truncation
        # does not discard its small negative-time/pre-ringing part.  The
        # delay is removed again when the valid output block is cropped.
        self.fir_delay_samples = self.impulse_length // 2
        requested_kernel_length = (
            self.impulse_fft_oversampling * self.impulse_length
        )
        self.impulse_fft_length = self._next_power_of_two_python(
            requested_kernel_length
        )

    @staticmethod
    def _next_power_of_two_python(value):
        value = int(value)
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _next_power_of_two_tensor(value):
        value = tf.cast(value, tf.float32)
        exponent = tf.math.ceil(tf.math.log(value) / tf.math.log(tf.constant(2.0)))
        return tf.cast(tf.pow(tf.constant(2.0), exponent), tf.int32)

    def fir_impulse_response(self):
        """Return the differentiable delayed finite impulse response."""
        frequency = tf.linspace(
            tf.constant(0.0, tf.float32),
            tf.cast(self.Fs / 2.0, tf.float32),
            self.impulse_fft_length // 2 + 1,
        )
        response = self.frequency_response(frequency)

        # A D-sample delay in frequency moves the band-limited impulse into
        # the finite FIR window.  Cropping at guard+D below cancels this delay.
        discrete_omega = (
            tf.cast(2.0 * np.pi / self.Fs, tf.float32) * frequency
        )
        phase = tf.complex(
            tf.zeros_like(discrete_omega),
            -discrete_omega * tf.cast(self.fir_delay_samples, tf.float32),
        )
        delayed_response = response * tf.exp(phase)
        periodic_impulse = tf.signal.irfft(
            delayed_response,
            fft_length=[self.impulse_fft_length],
        )
        return periodic_impulse[:self.impulse_length]

    def impulse_tail_energy_ratio(self, tail_samples=None):
        """Diagnostic ratio for detecting an impulse window that is too short."""
        impulse = self.fir_impulse_response()
        if tail_samples is None:
            tail_samples = max(1, self.impulse_length // 8)
        tail_samples = int(tail_samples)
        if not 1 <= tail_samples <= self.impulse_length:
            raise ValueError("tail_samples must be within the impulse length")
        total = tf.reduce_sum(tf.square(impulse))
        tail = tf.reduce_sum(tf.square(impulse[-tail_samples:]))
        return tail / (total + tf.keras.backend.epsilon())

    def call(self, inputs, training=None):
        rank = inputs.shape.rank
        if rank == 3:
            if inputs.shape[-1] not in (1, None):
                raise ValueError(
                    "CTLELayer_model_v12_v2 expects a singleton channel dimension"
                )
            waveform = tf.squeeze(inputs, axis=-1)
        elif rank == 2:
            waveform = inputs
        else:
            raise ValueError(
                "CTLELayer_model_v12_v2 expects [batch,time] or [batch,time,1]"
            )

        waveform = tf.cast(waveform, tf.float32)
        original_length = tf.shape(waveform)[1]
        tf.debugging.assert_greater_equal(
            original_length,
            self.guard_samples,
            message=(
                "CTLELayer_model_v12_v2 input must contain at least "
                "guard_samples samples for circular context"
            ),
        )

        guard = self.guard_samples
        circular_input = tf.concat(
            [waveform[:, -guard:], waveform, waveform[:, :guard]],
            axis=1,
        )
        impulse = self.fir_impulse_response()

        # Padding both operands to at least Nx+Nh-1 makes the FFT product an
        # exact linear convolution of circular_input and the finite FIR.
        full_length = tf.shape(circular_input)[1] + self.impulse_length - 1
        fft_length = self._next_power_of_two_tensor(full_length)
        fft_shape = tf.reshape(fft_length, [1])
        input_spectrum = tf.signal.rfft(circular_input, fft_length=fft_shape)
        impulse_spectrum = tf.signal.rfft(impulse, fft_length=fft_shape)
        full_output = tf.signal.irfft(
            input_spectrum * impulse_spectrum[tf.newaxis, :],
            fft_length=fft_shape,
        )

        # The central block begins after the circular history.  Advance the
        # crop by D to cancel the artificial FIR construction delay.
        start = tf.cast(guard + self.fir_delay_samples, tf.int32)
        output = tf.slice(
            full_output,
            begin=tf.stack([tf.constant(0, tf.int32), start]),
            size=tf.stack([tf.constant(-1, tf.int32), original_length]),
        )
        output.set_shape(waveform.shape)
        return output[:, :, tf.newaxis] if rank == 3 else output

    def get_config(self):
        config = super().get_config()
        config.update({
            "guard_time_s": self.guard_time_s,
            "guard_samples": self.guard_samples,
            "impulse_length": self.impulse_length,
            "impulse_fft_oversampling": self.impulse_fft_oversampling,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model_v12_v3(CTLELayer_model_v12):
    """v12 physical CTLE realized as a differentiable causal IIR cascade.

    No FFT and no frequency prewarping are used.  The physical v12 transfer
    function is separated into its second-order nodal core and first-order
    transport pole.  Standard Tustin substitution, s=2*Fs*(1-z^-1)/(1+z^-1),
    produces a biquad followed by a first-order section.

    A circular prefix is filtered before the requested block to warm up the
    recursive state.  With the Model_ch_Post rate of 100 GSa/s, the default
    10.24 ns warm-up corresponds to 1024 samples.  Set warmup_samples=0 for
    the zero-initial-state behavior of scipy.signal.lfilter/MATLAB filter.
    """

    def __init__(
        self,
        Fs=100e9,
        warmup_time_s=10.24e-9,
        warmup_samples=None,
        **kwargs,
    ):
        super().__init__(Fs=Fs, **kwargs)
        self.warmup_time_s = float(warmup_time_s)
        if not np.isfinite(self.warmup_time_s) or self.warmup_time_s < 0.0:
            raise ValueError("warmup_time_s must be finite and non-negative")
        automatic_warmup = int(np.ceil(self.Fs * self.warmup_time_s))
        self.warmup_samples = (
            automatic_warmup if warmup_samples is None else int(warmup_samples)
        )
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative")

    def analog_polynomial_coefficients(self):
        """Return ascending-s coefficients of the v12 nodal core and wt."""
        p = self.physical_parameters()
        op = self.operating_point_parameters()
        k_gm, k_gds, k_cgd, k_cgs, k_cdb, k_cds, k_ft = tf.unstack(
            tf.constant(self._calibration_np, dtype=tf.float32)
        )
        gm = k_gm * op["gm_S"]
        gds = k_gds * op["gds_S"]
        cgg = op["cgg_F"]
        cgd = k_cgd * op["cgd_F"]
        cgs = k_cgs * op["cgs_F"]
        cdb = k_cdb * op["cdb_F"]
        cds = k_cds * cgg

        # Each pair is the ascending polynomial p0 + p1*s.
        mdd0 = 1.0 / p["RL"] + gds
        mdd1 = p["CL"] + cdb + cgd + cds
        mds0 = -(gm + gds)
        mds1 = -cds
        msd0 = -gds
        msd1 = -cds
        mss0 = 2.0 / p["RS"] + gm + gds
        mss1 = 2.0 * p["CS"] + cgs + cds
        bd0 = -gm
        bd1 = cgd
        bs0 = gm
        bs1 = cgs

        numerator = tf.stack([
            bd0 * mss0 - mds0 * bs0,
            bd0 * mss1 + bd1 * mss0 - (mds0 * bs1 + mds1 * bs0),
            bd1 * mss1 - mds1 * bs1,
        ])
        denominator = tf.stack([
            mdd0 * mss0 - mds0 * msd0,
            mdd0 * mss1 + mdd1 * mss0 - (mds0 * msd1 + mds1 * msd0),
            mdd1 * mss1 - mds1 * msd1,
        ])
        omega_transport = k_ft * gm / cgg
        return numerator, denominator, omega_transport

    @staticmethod
    def _tustin_quadratic(coefficients, K):
        """Map p0+p1*s+p2*s^2 to z^-1 numerator coefficients."""
        p0, p1, p2 = tf.unstack(coefficients)
        K2 = K * K
        return tf.stack([
            p0 + p1 * K + p2 * K2,
            2.0 * p0 - 2.0 * p2 * K2,
            p0 - p1 * K + p2 * K2,
        ])

    def digital_iir_coefficients(self):
        """Return normalized (biquad_b, biquad_a, pole_b, pole_a)."""
        K = tf.cast(2.0 * self.Fs, tf.float32)
        return self._digital_iir_coefficients_for_K(K)

    def _digital_iir_coefficients_for_K(self, K):
        """Build the two digital sections for a supplied Tustin constant."""
        numerator_s, denominator_s, omega_transport = (
            self.analog_polynomial_coefficients()
        )
        K = tf.cast(K, tf.float32)
        numerator_z = self._tustin_quadratic(numerator_s, K)
        denominator_z = self._tustin_quadratic(denominator_s, K)
        core_b = numerator_z / denominator_z[0]
        core_a = denominator_z / denominator_z[0]

        # 1/(1+s/wt) -> (1+z^-1) /
        #                    ((1+K/wt)+(1-K/wt)z^-1)
        ratio = K / omega_transport
        transport_d0 = 1.0 + ratio
        transport_b = tf.stack([1.0 / transport_d0, 1.0 / transport_d0])
        transport_a = tf.stack([
            tf.constant(1.0, tf.float32),
            (1.0 - ratio) / transport_d0,
        ])
        return core_b, core_a, transport_b, transport_a

    @staticmethod
    def _filter_biquad(waveform, b, a):
        """Causal Direct Form I recurrence for a batch of waveforms."""
        samples = tf.transpose(waveform, [1, 0])
        zeros = tf.zeros_like(samples[0])
        b0, b1, b2 = tf.unstack(b)
        _, a1, a2 = tf.unstack(a)

        def step(states, current):
            x1, x2, y1, y2 = states
            output = (
                b0 * current + b1 * x1 + b2 * x2
                - a1 * y1 - a2 * y2
            )
            return current, x1, output, y1

        states = tf.scan(
            step,
            samples,
            initializer=(zeros, zeros, zeros, zeros),
        )
        return tf.transpose(states[2], [1, 0])

    @staticmethod
    def _filter_first_order(waveform, b, a):
        """Causal first-order Direct Form I recurrence."""
        samples = tf.transpose(waveform, [1, 0])
        zeros = tf.zeros_like(samples[0])
        b0, b1 = tf.unstack(b)
        _, a1 = tf.unstack(a)

        def step(states, current):
            x1, y1 = states
            output = b0 * current + b1 * x1 - a1 * y1
            return current, output

        states = tf.scan(
            step,
            samples,
            initializer=(zeros, zeros),
        )
        return tf.transpose(states[1], [1, 0])

    def discrete_poles(self):
        """Return differentiable coefficient arrays for external stability QA."""
        _, core_a, _, transport_a = self.digital_iir_coefficients()
        return core_a, transport_a

    def call(self, inputs, training=None):
        rank = inputs.shape.rank
        if rank == 3:
            if inputs.shape[-1] not in (1, None):
                raise ValueError(
                    "CTLELayer_model_v12_v3 expects a singleton channel dimension"
                )
            waveform = tf.squeeze(inputs, axis=-1)
        elif rank == 2:
            waveform = inputs
        else:
            raise ValueError(
                "CTLELayer_model_v12_v3 expects [batch,time] or [batch,time,1]"
            )

        waveform = tf.cast(waveform, tf.float32)
        original_length = tf.shape(waveform)[1]
        if self.warmup_samples:
            tf.debugging.assert_greater_equal(
                original_length,
                self.warmup_samples,
                message=(
                    "CTLELayer_model_v12_v3 input must contain at least "
                    "warmup_samples samples for circular state warm-up"
                ),
            )
            filtered_input = tf.concat(
                [waveform[:, -self.warmup_samples:], waveform], axis=1
            )
        else:
            filtered_input = waveform

        core_b, core_a, transport_b, transport_a = (
            self.digital_iir_coefficients()
        )
        output = self._filter_biquad(filtered_input, core_b, core_a)
        output = self._filter_first_order(output, transport_b, transport_a)
        if self.warmup_samples:
            output = output[:, self.warmup_samples:]
        output.set_shape(waveform.shape)
        return output[:, :, tf.newaxis] if rank == 3 else output

    def get_config(self):
        config = super().get_config()
        config.update({
            "warmup_time_s": self.warmup_time_s,
            "warmup_samples": self.warmup_samples,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CTLE")
class CTLELayer_model_v12_v4(CTLELayer_model_v12_v3):
    """v12 IIR with single-frequency prewarped Tustin discretization.

    The default prewarp point is 12.5 GHz, the first-Nyquist signal frequency
    of the current 25-GBd PAM4 link.  The continuous and discrete responses
    therefore agree exactly at that frequency (apart from float precision).
    No independent pole/zero-by-pole prewarping is applied in this version.
    """

    def __init__(
        self,
        Fs=100e9,
        prewarp_frequency_hz=12.5e9,
        **kwargs,
    ):
        super().__init__(Fs=Fs, **kwargs)
        self.prewarp_frequency_hz = float(prewarp_frequency_hz)
        if (
            not np.isfinite(self.prewarp_frequency_hz)
            or self.prewarp_frequency_hz <= 0.0
            or self.prewarp_frequency_hz >= self.Fs / 2.0
        ):
            raise ValueError(
                "prewarp_frequency_hz must be finite and strictly between "
                "0 and Fs/2"
            )

    def prewarped_tustin_constant(self):
        omega = tf.cast(2.0 * np.pi * self.prewarp_frequency_hz, tf.float32)
        half_discrete_angle = tf.cast(
            np.pi * self.prewarp_frequency_hz / self.Fs, tf.float32
        )
        return omega / tf.tan(half_discrete_angle)

    def digital_iir_coefficients(self):
        return self._digital_iir_coefficients_for_K(
            self.prewarped_tustin_constant()
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "prewarp_frequency_hz": self.prewarp_frequency_hz,
        })
        return config

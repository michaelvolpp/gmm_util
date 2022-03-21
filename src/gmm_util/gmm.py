import tensorflow as tf
import tensorflow_probability as tfp

from gmm_util.util import (
    prec_to_prec_tril,
    prec_to_scale_tril,
    scale_tril_to_cov,
    sample_gmm,
    gmm_log_density_grad_hess,
    gmm_log_component_densities,
    gmm_log_density_and_log_component_densities,
    gmm_log_responsibilities,
)


class GMM:
    def __init__(self, log_w: tf.Tensor, loc: tf.Tensor, prec: tf.Tensor):
        # check input
        self.n_components = log_w.shape[-1]
        self.d_z = loc.shape[-1]
        assert log_w.shape[-1:] == (self.n_components,)
        assert loc.shape[-2:] == (self.n_components, self.d_z)
        assert prec.shape[-3:] == (self.n_components, self.d_z, self.d_z)

        self._log_w = tf.Variable(tf.cast(log_w, dtype=tf.float32))
        self._loc = tf.Variable(tf.cast(loc, dtype=tf.float32))
        self._prec = tf.Variable(tf.cast(prec, dtype=tf.float32))
        self._prec_tril = tf.Variable(prec_to_prec_tril(prec=self._prec))
        self._scale_tril = tf.Variable(prec_to_scale_tril(prec=self._prec))
        self._cov = tf.Variable(scale_tril_to_cov(self._scale_tril))

    @property
    def loc(self):
        return self._loc

    @property
    def log_w(self):
        return self._log_w

    @property
    def prec(self):
        return self._prec

    @property
    def prec_tril(self):
        return self._prec_tril

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def cov(self):
        return self._cov

    # TODO: check consistency of shapes
    @loc.setter
    def loc(self, value):
        self._loc.assign(value)

    @log_w.setter
    def log_w(self, value):
        self._log_w.assign(value)

    @prec.setter
    def prec(self, value):
        self._prec.assign(value)
        self._prec_tril.assign(prec_to_prec_tril(self._prec))
        self._scale_tril.assign(prec_to_scale_tril(self.prec))
        self._cov.assign(scale_tril_to_cov(self.scale_tril))

    @prec_tril.setter
    def prec_tril(self, value):
        raise NotImplementedError

    @scale_tril.setter
    def scale_tril(self, value):
        raise NotImplementedError

    @cov.setter
    def cov(self, value):
        raise NotImplementedError

    def sample(self, n_samples: int):
        return sample_gmm(
            n_samples=n_samples,
            log_w=self.log_w,
            loc=self.loc,
            scale_tril=self.scale_tril,
        )

    def log_density(
        self,
        z: tf.Tensor,
        compute_grad: bool = False,
        compute_hess: bool = False,
    ):
        return gmm_log_density_grad_hess(
            z=z,
            log_w=self.log_w,
            loc=self.loc,
            prec=self.prec,
            scale_tril=self.scale_tril,
            compute_grad=compute_grad,
            compute_hess=compute_hess,
        )

    def log_component_densities(self, z: tf.Tensor):
        return gmm_log_component_densities(
            z=z, loc=self.loc, scale_tril=self.scale_tril
        )

    def log_responsibilities(self, z: tf.Tensor):
        log_dens, log_comp_dens = gmm_log_density_and_log_component_densities(
            z=z, log_w=self.log_w, loc=self.loc, scale_tril=self.scale_tril
        )
        return gmm_log_responsibilities(
            z=z,
            log_w=self.log_w,
            log_component_densities=log_comp_dens,
            log_density=log_dens,
        )

    def sample_all_components(self, num_samples_per_component):
        """
        Draws num_samples_per_component from each Gaussian component of every GMM in this model.
        """
        # TODO: is this efficient?
        samples = tfp.distributions.MultivariateNormalTriL(
            loc=self.loc, scale_tril=self.scale_tril, validate_args=False
        ).sample(num_samples_per_component)
        return samples

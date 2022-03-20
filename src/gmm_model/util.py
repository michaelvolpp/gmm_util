import math

import tensorflow as tf
import tensorflow_probability as tfp


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)]
)
def prec_to_prec_tril(prec: tf.Tensor):
    # Compute lower cholesky factors of precision matrices

    # check input
    if tf.executing_eagerly():
        d_z = prec.shape[-1]
        assert prec.shape[-2:] == (d_z, d_z)

    prec_tril = tf.linalg.cholesky(prec)

    # check output
    if tf.executing_eagerly():
        assert prec_tril.shape[-2:] == (d_z, d_z)
    return prec_tril


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)]
)
def prec_to_scale_tril(prec: tf.Tensor):
    # Compute lower cholesky factors of covariance matrices from precision matrices

    # check input
    if tf.executing_eagerly():
        d_z = prec.shape[-1]
        assert prec.shape[-2:] == (d_z, d_z)
    else:
        d_z = tf.shape(prec)[-1]

    # from: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
    Lf = tf.linalg.cholesky(tf.reverse(prec, axis=(-2, -1)))
    L_inv = tf.linalg.matrix_transpose(tf.reverse(Lf, axis=(-2, -1)))
    scale_tril = tf.linalg.triangular_solve(
        L_inv, tf.eye(d_z, dtype=L_inv.dtype), lower=True
    )

    # check output
    if tf.executing_eagerly():
        assert scale_tril.shape[-2:] == (d_z, d_z)
    return scale_tril


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)]
)
def scale_tril_to_cov(scale_tril: tf.Tensor):
    # check input
    if tf.executing_eagerly():
        d_z = scale_tril.shape[-1]
        assert scale_tril.shape[-2:] == (d_z, d_z)

    # compute cov from scale_tril
    cov = tf.matmul(scale_tril, tf.linalg.matrix_transpose(scale_tril))

    # check output
    if tf.executing_eagerly():
        assert cov.shape[-2:] == (d_z, d_z)
    return cov


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)]
)
def cov_to_scale_tril(cov: tf.Tensor):
    # check input
    if tf.executing_eagerly():
        d_z = cov.shape[-1]
        assert cov.shape[-2:] == (d_z, d_z)

    # compute scale_tril from cov
    scale_tril = tf.linalg.cholesky(cov)

    # check output
    if tf.executing_eagerly():
        assert scale_tril.shape[-2:] == (d_z, d_z)
    return scale_tril


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def sample_gmm(n_samples: int, log_w: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor):
    # check input
    if tf.executing_eagerly():
        d_z = loc.shape[-1]
        n_components = loc.shape[-2]
        batch_shape = loc.shape[:-2]
        assert log_w.shape == batch_shape + (n_components,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)

    # sample gmm
    # TODO: this is quite slow
    samples = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w, validate_args=False
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=False
        ),
        validate_args=False,
    ).sample(n_samples)

    # check output
    if tf.executing_eagerly():
        assert samples.shape == (n_samples,) + batch_shape + (d_z,)
    return samples


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def gmm_log_component_densities(z: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        batch_shape = z.shape[1:-1]
        n_components = loc.shape[-2]
        d_z = z.shape[-1]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)
    else:
        d_z = tf.shape(z)[-1]

    # compute log component densities
    diffs = z[..., None, :] - loc[None, ...]
    # assert diffs.shape == (n_samples,) + batch_shape + (n_components, d_z)
    # TODO: can we get away w/o broadcast_to by using sample-dim as last dim of diffs?
    sqrts = tf.linalg.triangular_solve(
        tf.broadcast_to(
            scale_tril[None, ...],
            tf.concat([tf.shape(diffs), [d_z]], axis=0),
        ),
        diffs[..., None],
    )
    sqrts = tf.squeeze(sqrts, axis=-1)
    # assert sqrts.shape == (n_samples,) + batch_shape + (n_components, d_z)
    mahalas = -0.5 * tf.reduce_sum(sqrts * sqrts, axis=-1)
    # assert mahalas.shape == (n_samples,) + batch_shape + (n_components,)
    const_parts = -0.5 * tf.reduce_sum(
        tf.math.log(tf.square(tf.linalg.diag_part(scale_tril))), axis=-1
    ) - 0.5 * tf.cast(d_z, tf.float32) * tf.math.log(2 * math.pi)
    # assert const_parts.shape == batch_shape + (n_components,)
    log_component_densities = mahalas + const_parts[None, ...]

    # check output
    if tf.executing_eagerly():
        assert log_component_densities.shape == (n_samples,) + batch_shape + (
            n_components,
        )
    return log_component_densities


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def gmm_log_density(
    z: tf.Tensor,
    log_w: tf.Tensor,
    log_component_densities: tf.Tensor,
):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = z.shape[-1]
        n_components = log_w.shape[-1]
        batch_shape = log_w.shape[:-1]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_w.shape == batch_shape + (n_components,)
        assert log_component_densities.shape == (n_samples,) + batch_shape + (
            n_components,
        )

    # compute log density
    log_joint_densities = log_component_densities + log_w[None, ...]
    # assert log_joint_densities.shape == (n_samples,) + batch_shape + (n_components,)
    log_density = tf.reduce_logsumexp(log_joint_densities, axis=-1)

    # check output
    if tf.executing_eagerly():
        assert log_density.shape == (n_samples,) + batch_shape
    return log_density


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    ]
)
def gmm_log_responsibilities(
    z: tf.Tensor,
    log_w: tf.Tensor,
    log_component_densities: tf.Tensor,
    log_density: tf.Tensor,
):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = z.shape[-1]
        n_components = log_w.shape[-1]
        batch_shape = log_w.shape[:-1]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_w.shape == batch_shape + (n_components,)
        assert log_density.shape == (n_samples,) + batch_shape
        assert log_component_densities.shape == (n_samples,) + batch_shape + (
            n_components,
        )

    # compute log density and responsibilities
    log_joint_densities = log_component_densities + log_w[None, ...]
    # assert log_joint_densities.shape == (n_samples,) + batch_shape + (n_components,)
    log_responsibilities = log_joint_densities - log_density[..., None]

    # check output
    if tf.executing_eagerly():
        assert log_responsibilities.shape == (n_samples,) + batch_shape + (
            n_components,
        )
    return log_responsibilities


def gmm_log_density_grad_hess(
    z: tf.Tensor,
    log_w: tf.Tensor,
    loc: tf.Tensor,
    prec: tf.Tensor,  # TODO: unify usage of prec, cov, scale_tril
    scale_tril: tf.Tensor,  # TODO: unify usage of prec, cov, scale_tril
    compute_grad: bool,
    compute_hess: bool,
):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = loc.shape[-1]
        n_components = loc.shape[-2]
        batch_shape = loc.shape[:-2]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_w.shape == batch_shape + (n_components,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert prec.shape == batch_shape + (n_components, d_z, d_z)
        assert not (compute_hess and not compute_grad)

    # compute log_density, and log_responsibilities
    log_density, log_component_densities = gmm_log_density_and_log_component_densities(
        z=z, log_w=log_w, loc=loc, scale_tril=scale_tril
    )

    # compute gradient
    if compute_grad:
        log_density_grad, log_responsibilities, prec_times_diff = gmm_log_density_grad(
            z=z,
            log_w=log_w,
            loc=loc,
            prec=prec,
            log_density=log_density,
            log_component_densities=log_component_densities,
        )
    else:
        log_density_grad = None

    # compute Hessian
    if compute_hess:
        log_density_hess = gmm_log_density_hess(
            prec=prec,
            log_responsibilities=log_responsibilities,
            prec_times_diff=prec_times_diff,
            log_density_grad=log_density_grad,
        )
    else:
        log_density_hess = None

    # check output
    if tf.executing_eagerly():
        assert log_density.shape == (n_samples,) + batch_shape
        if compute_grad:
            assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
        if compute_hess:
            assert log_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    return log_density, log_density_grad, log_density_hess


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def gmm_log_density_and_log_component_densities(
    z: tf.Tensor, log_w: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor
):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = loc.shape[-1]
        n_components = loc.shape[-2]
        batch_shape = loc.shape[:-2]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_w.shape == batch_shape + (n_components,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)

    # compute log component densities and log density
    log_comp_dens = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    log_dens = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=log_comp_dens,
    )

    # check output
    if tf.executing_eagerly():
        assert log_dens.shape == (n_samples,) + batch_shape
        assert log_comp_dens.shape == (n_samples,) + batch_shape + (n_components,)
    return log_dens, log_comp_dens


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def gmm_log_density_grad(
    z: tf.Tensor,
    log_w: tf.Tensor,
    loc: tf.Tensor,
    prec: tf.Tensor,
    log_density: tf.Tensor,
    log_component_densities: tf.Tensor,
):
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = loc.shape[-1]
        n_components = loc.shape[-2]
        batch_shape = loc.shape[:-2]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_w.shape == batch_shape + (n_components,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert prec.shape == batch_shape + (n_components, d_z, d_z)
        assert log_density.shape == (n_samples,) + batch_shape
        assert log_component_densities.shape == (n_samples,) + batch_shape + (
            n_components,
        )

    # compute d/dz log q(z)
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_density=log_density,
        log_component_densities=log_component_densities,
    )
    # assert log_responsibilities.shape == (n_samples,) + batch_shape + (n_components,)
    prec_times_diff = loc[None, ...] - z[..., None, :]
    # assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    prec_times_diff = tf.einsum("...ij,s...j->s...i", prec, prec_times_diff)
    # assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    # sum over components
    log_density_grad, signs = tfp.math.reduce_weighted_logsumexp(
        logx=log_responsibilities[..., None]
        + tf.math.log(tf.math.abs(prec_times_diff)),
        w=tf.math.sign(prec_times_diff),
        axis=-2,
        return_sign=True,
    )
    log_density_grad = tf.math.exp(log_density_grad) * signs

    # check output
    if tf.executing_eagerly():
        assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
        assert log_responsibilities.shape == (n_samples,) + batch_shape + (
            n_components,
        )
        assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    return log_density_grad, log_responsibilities, prec_times_diff


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def gmm_log_density_hess(
    prec: tf.Tensor,
    log_responsibilities: tf.Tensor,
    prec_times_diff: tf.Tensor,
    log_density_grad: tf.Tensor,
):
    # check input
    if tf.executing_eagerly():
        n_samples = log_responsibilities.shape[0]
        n_components = log_responsibilities.shape[-1]
        batch_shape = log_responsibilities.shape[1:-1]
        d_z = log_density_grad.shape[-1]
        assert prec.shape == batch_shape + (n_components, d_z, d_z)
        assert log_responsibilities.shape == (n_samples,) + batch_shape + (
            n_components,
        )
        assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
        assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)

    # compute Hessian
    log_density_hess = prec_times_diff - log_density_grad[..., None, :]
    # assert log_density_hess.shape == (n_samples,) + batch_shape + (n_components, d_z)
    log_density_hess = tf.einsum(
        "s...ki,s...kj->s...kij", log_density_hess, prec_times_diff
    )
    # assert log_density_hess.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    log_density_hess = log_density_hess - prec[None, ...]
    # assert log_density_hess.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    # sum over components
    log_density_hess, signs = tfp.math.reduce_weighted_logsumexp(
        logx=log_responsibilities[..., None, None]
        + tf.math.log(tf.math.abs(log_density_hess)),
        w=tf.math.sign(log_density_hess),
        axis=-3,
        return_sign=True,
    )
    log_density_hess = tf.math.exp(log_density_hess) * signs

    # check output
    if tf.executing_eagerly():
        assert log_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    return log_density_hess


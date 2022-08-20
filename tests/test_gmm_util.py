import math
import tensorflow as tf
import tensorflow_probability as tfp
from gmm_util.util import (
    prec_to_prec_tril,
    prec_to_scale_tril,
    scale_tril_to_cov,
    cov_to_scale_tril,
    cov_to_prec,
    sample_gmm,
    gmm_log_density,
    gmm_log_component_densities,
    gmm_log_responsibilities,
    gmm_log_density_grad_hess,
)
from .util_to_test_autograd import eval_fn_grad_hess
from gmm_util.gmm import GMM


def test_prec_to_prec_tril():
    tf.config.run_functions_eagerly(True)

    # low D
    L_true = tf.constant(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ],
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # low D, additional batch-dim
    L_true = tf.constant(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # high D: requires 64-bit precision to pass test
    tf.random.set_seed(123)
    L_true = tf.stack(
        [
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)


def test_prec_to_scale_tril():
    tf.config.run_functions_eagerly(True)

    # low D
    L = tf.constant(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ]
    )
    cov = L @ tf.linalg.matrix_transpose(L)
    prec = tf.linalg.inv(cov)
    scale_tril = prec_to_scale_tril(prec=prec)
    assert tf.experimental.numpy.allclose(scale_tril, L)

    # low D, additional batch-dim
    L_true = tf.constant(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # high D: requires 64-bit precision to pass test
    tf.random.set_seed(123)
    L = tf.stack(
        [
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
        ]
    )
    cov = L @ tf.linalg.matrix_transpose(L)
    prec = tf.linalg.inv(cov)
    scale_tril = prec_to_scale_tril(prec=prec)
    assert tf.experimental.numpy.allclose(scale_tril, L)


def test_scale_tril_to_cov():
    tf.config.run_functions_eagerly(True)

    # check 1
    scale_tril = tf.constant(
        [
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
        ]
    )
    true_cov = tf.constant(
        [
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        ]
    )
    cov = scale_tril_to_cov(scale_tril=scale_tril)
    assert tf.experimental.numpy.allclose(cov, true_cov)

    # check 2 with additional batch dim
    scale_tril = tf.constant(
        [
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
        ]
    )
    true_cov = tf.constant(
        [
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
        ]
    )
    cov = scale_tril_to_cov(scale_tril=scale_tril)
    assert tf.experimental.numpy.allclose(cov, true_cov)


def test_cov_to_prec():
    tf.config.run_functions_eagerly(True)

    # check 1
    cov = tf.constant(
        [
            [[2.0, 1.0], [6.0, 4.0]],
            [[2.0, 1.0], [6.0, 4.0]],
            [[2.0, 1.0], [6.0, 4.0]],
        ]
    )
    true_prec = tf.constant(
        [
            [[2.0, -0.5], [-3.0, 1.0]],
            [[2.0, -0.5], [-3.0, 1.0]],
            [[2.0, -0.5], [-3.0, 1.0]],
        ]
    )
    prec = cov_to_prec(cov=cov)
    assert tf.experimental.numpy.allclose(prec, true_prec)

    # check 2: with additional batch dim
    cov = tf.constant(
        [
            [
                [[2.0, 1.0], [6.0, 4.0]],
                [[2.0, 1.0], [6.0, 4.0]],
                [[2.0, 1.0], [6.0, 4.0]],
            ],
            [
                [[2.0, 1.0], [6.0, 4.0]],
                [[2.0, 1.0], [6.0, 4.0]],
                [[2.0, 1.0], [6.0, 4.0]],
            ],
        ]
    )
    true_prec = tf.constant(
        [
            [
                [[2.0, -0.5], [-3.0, 1.0]],
                [[2.0, -0.5], [-3.0, 1.0]],
                [[2.0, -0.5], [-3.0, 1.0]],
            ],
            [
                [[2.0, -0.5], [-3.0, 1.0]],
                [[2.0, -0.5], [-3.0, 1.0]],
                [[2.0, -0.5], [-3.0, 1.0]],
            ],
        ]
    )
    prec = cov_to_prec(cov=cov)
    assert tf.experimental.numpy.allclose(prec, true_prec)


def test_cov_to_scale_tril():
    tf.config.run_functions_eagerly(True)

    # check 1
    cov = tf.constant(
        [
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        ]
    )
    true_scale_tril = tf.constant(
        [
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
        ]
    )
    scale_tril = cov_to_scale_tril(cov=cov)
    assert tf.experimental.numpy.allclose(scale_tril, true_scale_tril)

    # check 2: with additional batch dim
    cov = tf.constant(
        [
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
        ]
    )
    true_scale_tril = tf.constant(
        [
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
        ]
    )
    scale_tril = cov_to_scale_tril(cov=cov)
    assert tf.experimental.numpy.allclose(scale_tril, true_scale_tril)


def test_sample_gmm():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    log_w = tf.math.log([1.0])
    loc = tf.constant([[-1.0]])
    scale_tril = tf.constant([[[0.1]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant([[-1.0], [1.0]])
    scale_tril = tf.constant([[[0.2]], [[0.2]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    log_w = tf.math.log([1.0])
    loc = tf.constant([[-1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_batch = 3
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ],
    )
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 4: d_z == 2, n_components == 2, batch_dim
    d_z = 2
    n_samples = 10
    n_batch = 3
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, n_batch, d_z)


def test_gmm_log_density():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, 2))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples, n_batch)
    assert log_densities.shape == (n_samples, n_batch)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)


def test_gmm_log_component_densities():
    tf.config.run_functions_eagerly(True)

    # check 1: d_z == 1, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 1))
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 2))
    loc = tf.constant([[1.0, -1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-2.0, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 2))
    loc = tf.constant([[1.0, -1.0], [1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    n_batch = 2
    z = tf.random.normal((n_samples, 2, 2))
    loc = tf.constant(
        [
            [[1.0, -1.0], [1.0, 1.0]],
            [[2.0, -2.0], [2.0, 2.0]],
        ]
    )
    scale_tril = tf.constant(
        [
            [[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]],
            [[[0.3, 0.0], [-3.0, 2.0]], [[0.4, 0.0], [-1.0, 2.0]]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, :, None, :])
    assert true_log_component_densities.shape == (n_samples, n_batch, n_components)
    assert log_component_densities.shape == (n_samples, n_batch, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )


def test_gmm_log_responsibilities():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                validate_args=True,
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, 2))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_batch, n_components)
    assert log_responsibilities.shape == (n_samples, n_batch, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )


def test_gmm_log_density_grad_hess():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_components = 2
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    d_z = 2
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, d_z))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples, n_batch)
    assert log_density_grad.shape == (n_samples, n_batch, d_z)
    assert log_density_hess.shape == (n_samples, n_batch, d_z, d_z)
    assert true_log_density.shape == (n_samples, n_batch)
    assert true_log_density_grad.shape == (n_samples, n_batch, d_z)
    assert true_log_density_hess.shape == (n_samples, n_batch, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )


def test_gmm():
    tf.config.run_functions_eagerly(True)  # s.t. various batch dims are possible

    ## (1) n_batch_dims = 0
    # generate valid parameters
    # set 1
    log_w = tf.math.log(tf.constant([0.1, 0.3, 0.6]))
    loc = tf.constant(
        [
            [3.0, 4.0],
            [-1.0, 2.0],
            [-7.0, -1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [1.0, 0.0],
                [-0.5, 0.1],
            ],
            [
                [7.0, 0.0],
                [1.0, 7.0],
            ],
            [
                [0.2, 0.0],
                [-9.0, 9.0],
            ],
        ]
    )
    cov = scale_tril_to_cov(scale_tril)
    prec = cov_to_prec(cov)
    prec_tril = prec_to_prec_tril(prec)
    # set 2
    log_w2 = tf.math.log(tf.constant([0.5, 0.1, 0.4]))
    loc2 = tf.constant(
        [
            [1.0, 4.0],
            [1.0, 2.0],
            [-5.0, -1.0],
        ]
    )
    scale_tril2 = tf.constant(
        [
            [
                [2.0, 0.0],
                [-0.5, 1.1],
            ],
            [
                [1.0, 0.0],
                [2.0, 9.0],
            ],
            [
                [0.3, 0.0],
                [-9.0, 9.0],
            ],
        ]
    )
    cov2 = scale_tril_to_cov(scale_tril2)
    prec2 = cov_to_prec(cov2)
    prec_tril2 = prec_to_prec_tril(prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(cov2), scale_tril2)
    assert tf.experimental.numpy.allclose(tf.linalg.inv(cov2), prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(prec2), prec_tril2)
    # (i) Initialize with precision
    gmm = GMM(log_w=log_w, loc=loc, prec=prec)
    assert gmm.n_batch_dims == 0
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (ii) Set new parameters (prec)
    gmm.log_w = log_w2
    gmm.loc = loc2
    gmm.prec = prec2
    assert gmm.n_batch_dims == 0
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w2)
    assert tf.experimental.numpy.allclose(gmm.prec, prec2)
    assert tf.experimental.numpy.allclose(gmm.cov, cov2)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril2)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril2)
    # (iii) Set new parameters (scale_tril)
    gmm.log_w = log_w
    gmm.loc = loc
    gmm.scale_tril = scale_tril
    assert gmm.n_batch_dims == 0
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (iv) Initialize with scale_tril
    gmm = GMM(log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert gmm.n_batch_dims == 0
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (v) call all methods (validity of results is confirmed by the other tests)
    z = tf.random.normal((10, 2))
    s = gmm.sample(n_samples=10)
    assert s.shape == (10, 2)
    ld, ldg, ldh = gmm.log_density(z=z, compute_grad=True, compute_hess=True)
    assert ld.shape == (10,)
    assert ldg.shape == (10, 2)
    assert ldh.shape == (10, 2, 2)
    lcd = gmm.log_component_densities(z=z)
    assert lcd.shape == (10, 3)
    lr = gmm.log_responsibilities(z=z)
    assert lr.shape == (10, 3)
    s = gmm.sample_all_components(n_samples_per_component=10)
    assert s.shape == (10, 3, 2)

    ## (2) n_batch_dims = 1
    # generate valid parameters
    # set 1
    log_w = tf.math.log(tf.constant([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]))
    loc = tf.constant(
        [
            [
                [3.0, 4.0],
                [-1.0, 2.0],
                [-7.0, -1.0],
            ],
            [
                [3.0, 4.0],
                [-1.0, 2.0],
                [-7.0, -1.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [
                    [1.0, 0.0],
                    [-0.5, 0.1],
                ],
                [
                    [7.0, 0.0],
                    [1.0, 7.0],
                ],
                [
                    [0.2, 0.0],
                    [-9.0, 9.0],
                ],
            ],
            [
                [
                    [1.0, 0.0],
                    [-0.5, 0.1],
                ],
                [
                    [7.0, 0.0],
                    [1.0, 7.0],
                ],
                [
                    [0.2, 0.0],
                    [-9.0, 9.0],
                ],
            ],
        ]
    )
    cov = scale_tril_to_cov(scale_tril)
    prec = cov_to_prec(cov)
    prec_tril = prec_to_prec_tril(prec)
    # set 2
    log_w2 = tf.math.log(tf.constant([[0.5, 0.1, 0.4], [0.5, 0.1, 0.4]]))
    loc2 = tf.constant(
        [
            [
                [1.0, 4.0],
                [1.0, 2.0],
                [-5.0, -1.0],
            ],
            [
                [1.0, 4.0],
                [1.0, 2.0],
                [-5.0, -1.0],
            ],
        ]
    )
    scale_tril2 = tf.constant(
        [
            [
                [
                    [2.0, 0.0],
                    [-0.5, 1.1],
                ],
                [
                    [1.0, 0.0],
                    [2.0, 9.0],
                ],
                [
                    [0.3, 0.0],
                    [-9.0, 9.0],
                ],
            ],
            [
                [
                    [2.0, 0.0],
                    [-0.5, 1.1],
                ],
                [
                    [1.0, 0.0],
                    [2.0, 9.0],
                ],
                [
                    [0.3, 0.0],
                    [-9.0, 9.0],
                ],
            ],
        ]
    )
    cov2 = scale_tril_to_cov(scale_tril2)
    prec2 = cov_to_prec(cov2)
    prec_tril2 = prec_to_prec_tril(prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(cov2), scale_tril2)
    assert tf.experimental.numpy.allclose(tf.linalg.inv(cov2), prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(prec2), prec_tril2)
    # (i) Initialize with precision
    gmm = GMM(log_w=log_w, loc=loc, prec=prec)
    assert gmm.n_batch_dims == 1
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (ii) Set new parameters (prec)
    gmm.log_w = log_w2
    gmm.loc = loc2
    gmm.prec = prec2
    assert gmm.n_batch_dims == 1
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w2)
    assert tf.experimental.numpy.allclose(gmm.prec, prec2)
    assert tf.experimental.numpy.allclose(gmm.cov, cov2)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril2)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril2)
    # (iii) Set new parameters (scale_tril)
    gmm.log_w = log_w
    gmm.loc = loc
    gmm.scale_tril = scale_tril
    assert gmm.n_batch_dims == 1
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (iv) Initialize with scale_tril
    gmm = GMM(log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert gmm.n_batch_dims == 1
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (v) call all methods (validity of results is confirmed by the other tests)
    z = tf.random.normal((10, 2, 2))
    s = gmm.sample(n_samples=10)
    assert s.shape == (10, 2, 2)
    ld, ldg, ldh = gmm.log_density(z=z, compute_grad=True, compute_hess=True)
    assert ld.shape == (10, 2)
    assert ldg.shape == (10, 2, 2)
    assert ldh.shape == (10, 2, 2, 2)
    lcd = gmm.log_component_densities(z=z)
    assert lcd.shape == (10, 2, 3)
    lr = gmm.log_responsibilities(z=z)
    assert lr.shape == (10, 2, 3)
    s = gmm.sample_all_components(n_samples_per_component=10)
    assert s.shape == (10, 2, 3, 2)

    ## (3) n_batch_dims = 2
    # generate valid parameters
    # set 1
    log_w = tf.math.log(
        tf.constant(
            [
                [
                    [0.1, 0.3, 0.6],
                    [0.1, 0.3, 0.6],
                ],
                [
                    [0.1, 0.3, 0.6],
                    [0.1, 0.3, 0.6],
                ],
            ]
        )
    )
    loc = tf.constant(
        [
            [
                [
                    [3.0, 4.0],
                    [-1.0, 2.0],
                    [-7.0, -1.0],
                ],
                [
                    [3.0, 4.0],
                    [-1.0, 2.0],
                    [-7.0, -1.0],
                ],
            ],
            [
                [
                    [3.0, 4.0],
                    [-1.0, 2.0],
                    [-7.0, -1.0],
                ],
                [
                    [3.0, 4.0],
                    [-1.0, 2.0],
                    [-7.0, -1.0],
                ],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [
                    [
                        [1.0, 0.0],
                        [-0.5, 0.1],
                    ],
                    [
                        [7.0, 0.0],
                        [1.0, 7.0],
                    ],
                    [
                        [0.2, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
                [
                    [
                        [1.0, 0.0],
                        [-0.5, 0.1],
                    ],
                    [
                        [7.0, 0.0],
                        [1.0, 7.0],
                    ],
                    [
                        [0.2, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
            ],
            [
                [
                    [
                        [1.0, 0.0],
                        [-0.5, 0.1],
                    ],
                    [
                        [7.0, 0.0],
                        [1.0, 7.0],
                    ],
                    [
                        [0.2, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
                [
                    [
                        [1.0, 0.0],
                        [-0.5, 0.1],
                    ],
                    [
                        [7.0, 0.0],
                        [1.0, 7.0],
                    ],
                    [
                        [0.2, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
            ],
        ]
    )
    cov = scale_tril_to_cov(scale_tril)
    prec = cov_to_prec(cov)
    prec_tril = prec_to_prec_tril(prec)
    # set 2
    log_w2 = tf.math.log(
        tf.constant(
            [
                [
                    [0.5, 0.1, 0.4],
                    [0.5, 0.1, 0.4],
                ],
                [
                    [0.5, 0.1, 0.4],
                    [0.5, 0.1, 0.4],
                ],
            ]
        )
    )
    loc2 = tf.constant(
        [
            [
                [
                    [1.0, 4.0],
                    [1.0, 2.0],
                    [-5.0, -1.0],
                ],
                [
                    [1.0, 4.0],
                    [1.0, 2.0],
                    [-5.0, -1.0],
                ],
            ],
            [
                [
                    [1.0, 4.0],
                    [1.0, 2.0],
                    [-5.0, -1.0],
                ],
                [
                    [1.0, 4.0],
                    [1.0, 2.0],
                    [-5.0, -1.0],
                ],
            ],
        ]
    )
    scale_tril2 = tf.constant(
        [
            [
                [
                    [
                        [2.0, 0.0],
                        [-0.5, 1.1],
                    ],
                    [
                        [1.0, 0.0],
                        [2.0, 9.0],
                    ],
                    [
                        [0.3, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
                [
                    [
                        [2.0, 0.0],
                        [-0.5, 1.1],
                    ],
                    [
                        [1.0, 0.0],
                        [2.0, 9.0],
                    ],
                    [
                        [0.3, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
            ],
            [
                [
                    [
                        [2.0, 0.0],
                        [-0.5, 1.1],
                    ],
                    [
                        [1.0, 0.0],
                        [2.0, 9.0],
                    ],
                    [
                        [0.3, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
                [
                    [
                        [2.0, 0.0],
                        [-0.5, 1.1],
                    ],
                    [
                        [1.0, 0.0],
                        [2.0, 9.0],
                    ],
                    [
                        [0.3, 0.0],
                        [-9.0, 9.0],
                    ],
                ],
            ],
        ]
    )
    cov2 = scale_tril_to_cov(scale_tril2)
    prec2 = cov_to_prec(cov2)
    prec_tril2 = prec_to_prec_tril(prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(cov2), scale_tril2)
    assert tf.experimental.numpy.allclose(tf.linalg.inv(cov2), prec2)
    assert tf.experimental.numpy.allclose(tf.linalg.cholesky(prec2), prec_tril2)
    # (i) Initialize with precision
    gmm = GMM(log_w=log_w, loc=loc, prec=prec)
    assert gmm.n_batch_dims == 2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (ii) Set new parameters (prec)
    gmm.log_w = log_w2
    gmm.loc = loc2
    gmm.prec = prec2
    assert gmm.n_batch_dims == 2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w2)
    assert tf.experimental.numpy.allclose(gmm.prec, prec2)
    assert tf.experimental.numpy.allclose(gmm.cov, cov2)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril2)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril2)
    # (iii) Set new parameters (scale_tril)
    gmm.log_w = log_w
    gmm.loc = loc
    gmm.scale_tril = scale_tril
    assert gmm.n_batch_dims == 2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (iv) Initialize with scale_tril
    gmm = GMM(log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert gmm.n_batch_dims == 2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert tf.experimental.numpy.allclose(gmm.log_w, log_w)
    assert tf.experimental.numpy.allclose(gmm.prec, prec)
    assert tf.experimental.numpy.allclose(gmm.cov, cov)
    assert tf.experimental.numpy.allclose(gmm.prec_tril, prec_tril)
    assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
    # (v) call all methods (validity of results is confirmed by the other tests)
    z = tf.random.normal((10, 2, 2, 2))
    s = gmm.sample(n_samples=10)
    assert s.shape == (10, 2, 2, 2)
    ld, ldg, ldh = gmm.log_density(z=z, compute_grad=True, compute_hess=True)
    assert ld.shape == (10, 2, 2)
    assert ldg.shape == (10, 2, 2, 2)
    assert ldh.shape == (10, 2, 2, 2, 2)
    lcd = gmm.log_component_densities(z=z)
    assert lcd.shape == (10, 2, 2, 3)
    lr = gmm.log_responsibilities(z=z)
    assert lr.shape == (10, 2, 2, 3)
    s = gmm.sample_all_components(n_samples_per_component=10)
    assert s.shape == (10, 2, 2, 3, 2)

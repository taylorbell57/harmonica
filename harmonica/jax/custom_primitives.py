import jax
import numpy as np
import jax.numpy as jnp
from jaxlib import xla_client
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir
from jax._src.interpreters.mlir import ir, custom_call
from jaxlib.mlir.dialects import mhlo

from harmonica.core import bindings

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)


def ir_dtype(np_dtype):
    """Convert a NumPy dtype or Python type to an MLIR IR type."""
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float32:
        return ir.F32Type.get()
    elif np_dtype == np.float64:
        return ir.F64Type.get()
    elif np_dtype == np.int32:
        return ir.IntegerType.get_signless(32)
    elif np_dtype == np.int64:
        return ir.IntegerType.get_signless(64)
    else:
        raise TypeError(f"Unsupported dtype: {np_dtype}")


def ir_constant(val):
    attr = ir.DenseElementsAttr.get(
        np.array(val).reshape(()).astype(np.dtype(type(val)))
    )
    return mhlo.ConstantOp(attr).result


def _harmonica_transit_common(primitive_fn, times, params, r):
    """Internal helper function to broadcast inputs and call the JAX primitive.

    Parameters
    ----------
    primitive_fn : Callable
        The JAX primitive function to evaluate.
    times : array_like
        Scalar or 1D array of times at which to evaluate the model.
    params : list of scalars or array_like
        Orbital and limb-darkening parameters to be passed to the primitive.
    r : array_like
        Fourier coefficients specifying the transmission string geometry.

    Returns
    -------
    outputs : tuple
        Tuple of (flux, Jacobian), as returned by the primitive function.
    """
    # Convert to JAX arrays (allow scalars as 1D)
    times = jnp.atleast_1d(jnp.asarray(times))
    if times.ndim > 1:
        raise ValueError("`times` must be a scalar or 1D array")

    n = times.shape[0]

    # Broadcast scalar/1D params to shape (n,)
    broadcasted_params = []
    for p in params:
        p = jnp.asarray(p, dtype=jnp.float64)
        if p.ndim == 0:
            p = jnp.broadcast_to(p, (n,))
        elif p.ndim == 1 and p.shape[0] != n:
            p = jnp.broadcast_to(p, (n,))
        elif p.ndim != 1:
            raise ValueError(f"Unexpected parameter shape: {p.shape}")
        broadcasted_params.append(p)

    # Flip sign of flux if r[0] is negative (model upside-down transit)
    def shift_r0_if_needed(r_row):
        r0 = r_row[0]
        r_shifted = r_row.at[0].set(jnp.abs(r0))
        return jnp.where(r0 < 0, r_shifted, r_row)

    def replace_with_default_ripple(r_row):
        fallback = r_row.at[1].set(1e-3)
        if r_row.shape[0] > 3:
            fallback = fallback.at[3].set(5e-4)
        return fallback

    def enforce_min_ripple(r_row):
        if r_row.shape[0] < 2:
            return r_row
        ripple = r_row[1:]
        rms = jnp.sqrt(jnp.mean(ripple ** 2))
        return jax.lax.cond(rms < 1e-6, replace_with_default_ripple, lambda x: x, r_row)

    if r.ndim == 1:
        flip = r[0] < 0
        r = shift_r0_if_needed(r)
        r = enforce_min_ripple(r)
    elif r.ndim == 2:
        flip = r[:, 0] < 0
        r = jax.vmap(shift_r0_if_needed)(r)
        r = jax.vmap(enforce_min_ripple)(r)
    else:
        raise ValueError(f"`r` must be shape (k,) or (n, k); got {r.shape}")

    # Rebuild r_list after fixing r
    if r.ndim == 1:
        r_list = [jnp.broadcast_to(r[i], (n,)) for i in range(r.shape[0])]
    else:
        r_list = [jnp.reshape(r[:, i], (n,)) for i in range(r.shape[1])]

    # Combine all args and ensure all are float64 arrays.
    args = [jnp.asarray(arg, dtype=jnp.float64) for arg in
            (times, *broadcasted_params, *r_list)]

    flux, *rest = primitive_fn(*args)
    flux = jnp.where(flip, 2.0 - flux, flux)
    return (flux, *rest)


@jax.jit
def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1])):
    """ Harmonica transits with jax -- quadratic limb darkening.

    Parameters
    ----------
    times : ndarray
        1D array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2 : floats
        Quadratic limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].
    """
    return _harmonica_transit_common(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r
    )[0]


@jax.jit
def _quad_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                  u1, u2, r):
    """Return both the flux and its derivatives for the quadratic LD model.

    Intended for internal use and testing. This function bypasses the
    simplified public API and directly exposes both model outputs.

    Parameters
    ----------
    Same as `harmonica_transit_quad_ld`.

    Returns
    -------
    flux : ndarray
        Model flux values.
    d_flux_d_params : ndarray
        Derivatives of the flux with respect to all input parameters.
    """
    return _prepare_args_and_call_primitive(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r
    )


def jax_light_curve_quad_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_quad_ld_p.bind(times, *params)


def jax_light_curve_quad_ld_abstract_eval(abstract_times, *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_quad_ld_xla_translation(ctx, timesc, *paramssc):
    """ XLA compilation rules. """
    # Get `shape` info.
    timesc_shape = ctx.avals_in[0]

    # Define input `shapes`.
    data_type = timesc_shape.dtype
    shape = timesc_shape.shape

    # Additionally, define the number of model evaluation points.
    n_times = np.prod(shape).astype(np.int64)
    n_times_const = ir_constant(np.int64(n_times))

    # Additionally, define the number of transmission string coefficients.
    n_rs = len(paramssc) - 6 - 2
    n_rs_const = ir_constant(np.int64(n_rs))

    # Define output `shapes`.
    output_shape_model_eval = ir.RankedTensorType.get(
        shape, ir_dtype(data_type))
    shape_derivatives = shape + (6 + 2 + n_rs,)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_quad_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[
            (), (), list(reversed(range(len(shape))))
        ] + [
            list(reversed(range(len(shape))))
        ] * len(paramssc),
        result_layouts=[
            list(reversed(range(len(shape)))),
            list(reversed(range(len(shape_derivatives))))
        ]
    ).results


def jax_light_curve_quad_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    _, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_quad_ld_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if type(pd) is ad.Zero:
            # This partial derivative is not required. It has been
            # set to a deterministic value.
            continue
        df += pd * df_dz[..., idx_pd]

    # Return valid zero tangent for the second output (Jacobian) as we are
    # not interested in using it for gradient-based inference.
    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


@jax.jit
def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    """ Harmonica transits with jax -- non-linear limb darkening.

    Parameters
    ----------
    times : ndarray
        1D array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u1, u2, u3, u4 : floats
        Non-linear limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].
    """
    return _harmonica_transit_common(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r
    )[0]


@jax.jit
def _nonlinear_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                       u1, u2, u3, u4, r):
    """Return both the flux and its derivatives for the non-linear LD model.

    Intended for internal use and testing. This function bypasses the
    simplified public API and directly exposes both model outputs.

    Parameters
    ----------
    Same as `harmonica_transit_nonlinear_ld`.

    Returns
    -------
    flux : ndarray
        Model flux values.
    d_flux_d_params : ndarray
        Derivatives of the flux with respect to all input parameters.
    """
    f, df_dz = _prepare_args_and_call_primitive(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r
    )
    return f, df_dz


def jax_light_curve_nonlinear_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_nonlinear_ld_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_abstract_eval(abstract_times,
                                               *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_nonlinear_xla_translation(ctx, timesc, *paramssc):
    """MLIR lowering for nonlinear LD transit model."""
    timesc_shape = ctx.avals_in[0]
    data_type = timesc_shape.dtype
    shape = timesc_shape.shape

    n_times = np.prod(shape).astype(np.int64)
    n_times_const = ir_constant(n_times)

    n_rs = len(paramssc) - 6 - 4
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get(
        shape, ir_dtype(data_type))
    shape_derivatives = shape + (6 + 4 + n_rs,)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_nonlinear_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[
            (), (), list(reversed(range(len(shape))))
        ] + [
            list(reversed(range(len(shape))))
        ] * len(paramssc),
        result_layouts=[
            list(reversed(range(len(shape)))),
            list(reversed(range(len(shape_derivatives))))
        ]
    ).results


def jax_light_curve_nonlinear_ld_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    f, df_dz = jax_light_curve_nonlinear_ld_prim(times, *args)

    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if type(pd) is ad.Zero:
            continue
        df += pd * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


# Register the C++ models, bytes string required.
xla_client.register_custom_call_target(
    b'jax_light_curve_quad_ld',
    bindings.jax_registrations()['jax_light_curve_quad_ld']
)
xla_client.register_custom_call_target(
    b'jax_light_curve_nonlinear_ld',
    bindings.jax_registrations()['jax_light_curve_nonlinear_ld']
)


# Common utility function to prepare arguments for primitives
def _prepare_args_and_call_primitive(primitive_fn, times, param_list, r):
    return _harmonica_transit_common(primitive_fn, times, param_list, r)


# Create a primitive for quad ld.
jax_light_curve_quad_ld_p = Primitive('jax_light_curve_quad_ld')
jax_light_curve_quad_ld_p.multiple_results = True
jax_light_curve_quad_ld_p.def_abstract_eval(
    jax_light_curve_quad_ld_abstract_eval)
mlir.register_lowering(
    jax_light_curve_quad_ld_p, jax_light_curve_quad_ld_xla_translation,
    platform='cpu')
ad.primitive_jvps[jax_light_curve_quad_ld_p] = (
    jax_light_curve_quad_ld_value_and_jvp)

# Create a primitive for non-linear ld.
jax_light_curve_nonlinear_ld_p = Primitive(
    'jax_light_curve_nonlinear_ld')
jax_light_curve_nonlinear_ld_p.multiple_results = True
jax_light_curve_nonlinear_ld_p.def_abstract_eval(
    jax_light_curve_nonlinear_ld_abstract_eval)
mlir.register_lowering(
    jax_light_curve_nonlinear_ld_p, jax_light_curve_nonlinear_xla_translation,
    platform='cpu')
ad.primitive_jvps[jax_light_curve_nonlinear_ld_p] = (
    jax_light_curve_nonlinear_ld_value_and_jvp)

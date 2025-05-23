import jax
import numpy as np
import jax.numpy as jnp
from jaxlib import xla_client
from functools import partial
from jax.interpreters import ad, mlir
from jax._src.interpreters import mlir as jax_mlir
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
    dtype = ir_dtype(type(val))
    attr = ir.DenseElementsAttr.get(np.array(val).reshape(()).astype(np.dtype(type(val))))
    return mhlo.ConstantOp(attr).result


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
    # Unpack model parameters.
    params = [t0, period, a, inc, ecc, omega, u1, u2]
    for rn in r:
        params.append(rn)

    # Broadcast parameters to same length as times.
    times, *args_struc = jnp.broadcast_arrays(times, *params)

    return jax_light_curve_quad_ld_prim(times, *args_struc)[0]


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
    dims_order = tuple(range(len(shape) - 1, -1, -1))
    input_shape = ir.RankedTensorType.get(shape, ir_dtype(data_type))
    rs_input_shapes = [input_shape] * len(paramssc)

    # Additionally, define the number of model evaluation points.
    n_times = np.prod(shape).astype(np.int64)
    n_times_type = ir.RankedTensorType.get((), ir_dtype(np.int64))
    n_times_const = ir_constant(np.int64(n_times))

    # Additionally, define the number of transmission string coefficients.
    n_rs = len(paramssc) - 6 - 2
    n_rs_type = ir.RankedTensorType.get((), ir_dtype(np.int64))
    n_rs_const = ir_constant(np.int64(n_rs))

    # Define output `shapes`.
    output_shape_model_eval = ir.RankedTensorType.get(shape, ir_dtype(data_type))
    shape_derivatives = shape + (6 + 2 + n_rs,)
    output_shape_model_derivatives = ir.RankedTensorType.get(shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_quad_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[(), (), list(reversed(range(len(shape))))] + [list(reversed(range(len(shape))))] * len(paramssc),
        result_layouts=[list(reversed(range(len(shape)))), list(reversed(range(len(shape_derivatives))))]
   ).results


def jax_light_curve_quad_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    dtimes, *dargs = arg_tangents

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

    # None is returned here for the second output as we are not interested
    # in using it for gradient-based inference.
    return (f, df_dz), (df, None)


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
    # Unpack model parameters.
    params = [t0, period, a, inc, ecc, omega, u1, u2, u3, u4]
    for rn in r:
        params.append(rn)

    # Broadcast parameters to same length as times.
    times, *args_struc = jnp.broadcast_arrays(times, *params)

    return jax_light_curve_nonlinear_ld_prim(times, *args_struc)[0]


def jax_light_curve_nonlinear_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_nonlinear_ld_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_abstract_eval(abstract_times, *abstract_params):
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

    input_shape = ir.RankedTensorType.get(shape, ir_dtype(data_type))
    rs_input_shapes = [input_shape] * len(paramssc)

    n_times = np.prod(shape).astype(np.int64)
    n_times_const = ir_constant(n_times)

    n_rs = len(paramssc) - 6 - 4
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get(shape, ir_dtype(data_type))
    shape_derivatives = shape + (6 + 4 + n_rs,)
    output_shape_model_derivatives = ir.RankedTensorType.get(shape_derivatives, ir_dtype(data_type))

    return custom_call(
        b"jax_light_curve_nonlinear_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        operand_layouts=[(), (), list(reversed(range(len(shape))))] + [list(reversed(range(len(shape))))] * len(paramssc),
        result_layouts=[list(reversed(range(len(shape)))), list(reversed(range(len(shape_derivatives))))]
    ).results


def jax_light_curve_nonlinear_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    dtimes, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_nonlinear_ld_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if type(pd) is ad.Zero:
            # This partial derivative is not required. It has been
            # set to a deterministic value.
            continue
        df += pd * df_dz[..., idx_pd]

    # None is returned here for the second output as we are not interested
    # in using it for gradient-based inference.
    return (f, df_dz), (df, None)


# Register the C++ models, bytes string required.
xla_client.register_custom_call_target(
    b'jax_light_curve_quad_ld', bindings.jax_registrations()['jax_light_curve_quad_ld'])
xla_client.register_custom_call_target(
    b'jax_light_curve_nonlinear_ld', bindings.jax_registrations()['jax_light_curve_nonlinear_ld'])

# Create a primitive for quad ld.
jax_light_curve_quad_ld_p = jax.core.Primitive('jax_light_curve_quad_ld')
jax_light_curve_quad_ld_p.multiple_results = True
# jax_light_curve_quad_ld_p.def_impl(partial(xla.apply_primitive, jax_light_curve_quad_ld_p))
def impl_quad_ld(*args):
    return jax_light_curve_quad_ld_prim(*args)
# jax_light_curve_quad_ld_p.def_impl(impl_quad_ld)
jax_light_curve_quad_ld_p.def_abstract_eval(jax_light_curve_quad_ld_abstract_eval)
# xla.backend_specific_translations['cpu'][jax_light_curve_quad_ld_p] = \
#     jax_light_curve_quad_ld_xla_translation
mlir.register_lowering(jax_light_curve_quad_ld_p, jax_light_curve_quad_ld_xla_translation, platform='cpu')
ad.primitive_jvps[jax_light_curve_quad_ld_p] = jax_light_curve_quad_ld_value_and_jvp

# Create a primitive for non-linear ld.
jax_light_curve_nonlinear_ld_p = jax.core.Primitive('jax_light_curve_nonlinear_ld')
jax_light_curve_nonlinear_ld_p.multiple_results = True
# jax_light_curve_nonlinear_ld_p.def_impl(partial(xla.apply_primitive, jax_light_curve_nonlinear_ld_p))
def impl_nonlinear_ld(*args):
    return jax_light_curve_nonlinear_ld_prim(*args)
# jax_light_curve_nonlinear_ld_p.def_impl(impl_nonlinear_ld)
jax_light_curve_nonlinear_ld_p.def_abstract_eval(jax_light_curve_nonlinear_ld_abstract_eval)
# xla.backend_specific_translations['cpu'][jax_light_curve_nonlinear_ld_p] = \
#     jax_light_curve_nonlinear_xla_translation
mlir.register_lowering(jax_light_curve_nonlinear_ld_p, jax_light_curve_nonlinear_xla_translation, platform='cpu')
ad.primitive_jvps[jax_light_curve_nonlinear_ld_p] = jax_light_curve_nonlinear_ld_value_and_jvp

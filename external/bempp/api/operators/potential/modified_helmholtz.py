"""Modified Helmholtz potential operators."""
import numpy as _np


def grad_kernel(
    space,
    points,
    wavenumber=1,  
    epsilon=1,
    c = (0,0,0),
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz single-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler
    from .modified_helmholtz import single_layer_scaled as modified_single_layer_scaled

    
    if _np.real(wavenumber) == 0:
        return modified_single_layer_scaled(
            space,
            points,
            _np.imag(wavenumber),
            epsilon,
            c,
            parameters,
            assembler,
            device_interface,
            precision,
        )
    

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "mod_helmholtz_grad_single_layer_potential_scaled",  # Identifier
        [omega, epsilon, c[0], c[1], c[2]],  # Options
        "mod_helmholtz_grad_single_layer_scaled",  # Kernel type
        "default_vector",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def single_layer(
    space,
    points,
    omega,
    epsilon=1,
    c_k=(0,0,0), 
    c_l=(0,0,0),
    custom_mode=False,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a modified Helmholtz single-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "modified_helmholtz_single_layer_potential",  # Identifier
        [omega, epsilon, c_k[0], c_k[1], c_k[2], c_l[0], c_l[1], c_l[2], custom_mode],  # Options
        "modified_helmholtz_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def single_layer_scaled(
    space,
    points,
    omega,
    epsilon=1,
    c = (0,0,0),
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a modified Helmholtz single-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "modified_helmholtz_single_layer_potential_scaled",  # Identifier
        [omega, epsilon, c[0], c[1], c[2]],  # Options
        "modified_helmholtz_single_layer_scaled",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def double_layer(
    space,
    points,
    omega,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a modified Helmholtz double-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "modified_helmholtz_double_layer_potential",  # Identifier
        [omega],  # Options
        "modified_helmholtz_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

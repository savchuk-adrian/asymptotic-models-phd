"""Helmholtz potential operators."""
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
        "helmholtz_grad_single_layer_potential_scaled",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber), epsilon, c[0], c[1], c[2]],  # Options
        "helmholtz_grad_single_layer_scaled",  # Kernel type
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
    wavenumber,
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
    from .modified_helmholtz import single_layer as modified_single_layer

    if _np.real(wavenumber) == 0:
        return modified_single_layer(
            space,
            points,
            wavenumber,
            parameters,
            assembler,
            device_interface,
            precision,
        )

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_single_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
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
        "helmholtz_single_layer_potential_scaled",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber), epsilon, c[0], c[1], c[2]],  # Options
        "helmholtz_single_layer_scaled",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
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
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz double-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler
    from .modified_helmholtz import double_layer as modified_double_layer

    if _np.real(wavenumber) == 0:
        return modified_double_layer(
            space,
            points,
            wavenumber,
            parameters,
            assembler,
            device_interface,
            precision,
        )

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_double_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

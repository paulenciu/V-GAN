from src.vmmd.VMMDConvLinearMapping import VMMDConvLinearMapping
from src.vmmd.VMMDFlattened import VMMDFlattened
from src.vmmd.VMMDLinearMapping import VMMDLinearMapping
from src.vmmd.VMMDRotationMapping import VMMDRotationMapping
from src.vmmd.VMMDSingleMask import VMMDSingleMask


def load_vmmd(filepath, generator = None, name="single"):
    if name == "single":
        vmmd = VMMDSingleMask()
        vmmd.load_models(generator=generator, path_to_generator=filepath, ndims=1024)
    elif name == "flatten":
        vmmd = VMMDFlattened()
        vmmd.load_models(generator=generator, path_to_generator=filepath, ndims=3072)
    elif name == "linear":
        vmmd = VMMDLinearMapping()
        vmmd.load_models(generator=generator, path_to_generator=filepath, ndims=32)
    elif name == "rotation":
        vmmd = VMMDRotationMapping()
        vmmd.load_models(generator=generator, path_to_generator=filepath, ndims=2)
    elif name == "conv_linear":
        vmmd = VMMDConvLinearMapping()
        vmmd.load_models(generator=generator, path_to_generator=filepath, ndims=2)
    else:
        raise NotImplementedError("Error, generator with label not found.")
    return vmmd
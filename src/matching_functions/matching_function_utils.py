from . import DiagonalMatching, MeanMatching, OTAMMatching, MaxMatching
from . import LinearMatching, ChamferMatching


def get_matching_function(args):
    """Select the matching function based on the input arguments

    :param args: the input arguments
    :return: the matching function module
    """
    if args.matching_function == "diag":
        return DiagonalMatching(args)
    elif args.matching_function == "linear":
        return LinearMatching(args)
    elif args.matching_function == "mean":
        return MeanMatching(args)
    elif args.matching_function == "max":
        return MaxMatching(args)
    elif args.matching_function == "otam":
        return OTAMMatching(args)
    elif args.matching_function.startswith("chamfer"):
        return ChamferMatching(args)
    else:
        raise Exception("Not implemented yet")




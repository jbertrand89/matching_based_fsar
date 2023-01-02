from . import DiagonalMatching, MeanMatching, OTAMMatching, MaxMatching
from . import LinearMatching, ChamferMatching
#,, ,


def get_matching_function(args):
    if args.matching_function == "diag":
        return DiagonalMatching(args)
    elif args.matching_function == "fc":
        return LinearMatching(args)
    elif args.matching_function == "mean":
        return MeanMatching(args)
    elif args.matching_function == "max":
        return MaxMatching(args)
    elif args.matching_function == "otam":
        return OTAMMatching(args)
    elif args.matching_function in {"chamfer", "chamfer-transposed", "chamfer-support"}:
        return ChamferMatching(args)
    else:
        raise Exception("Not implemented yet")




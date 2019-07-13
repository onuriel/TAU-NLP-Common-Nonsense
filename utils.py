import argparse


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('{} is not a positive integer '.format(ivalue))
    return ivalue

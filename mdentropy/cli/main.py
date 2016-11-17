from __future__ import print_function, absolute_import, division

import sys
import argparse

from .. import __version__
from . import dmutinf
from . import dtent


def main():
    help = ('MDEntropy is a python library that allows users to perform '
            'information-theoretic analyses on molecular dynamics (MD) '
            'trajectories.')
    p = argparse.ArgumentParser(description=help)
    p.add_argument(
        '-V', '--version',
        action='version',
        version='mdentropy %s' % __version__,
    )
    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    dmutinf.configure_parser(sub_parsers)
    dtent.configure_parser(sub_parsers)

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = p.parse_args()
    args_func(args, p)


def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred with mdentropy (version %s), please
consider sending the following traceback to the mdentropy GitHub issue tracker at:
        https://github.com/msmbuilder/mdentropy/issues
"""
            print(message % __version__, file=sys.stderr)
        raise  # as if we did not catch it

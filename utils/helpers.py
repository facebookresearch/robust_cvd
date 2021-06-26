#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import atexit
import io
import os
import shutil
import sys
import tempfile

from utils import frame_range


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def mkdir_ifnotexists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def print_title(text):
    print()
    print("-" * len(text))
    print(text)
    print("-" * len(text))
    print()


def print_banner(text):
    w = 12 + len(text)
    print()
    print("*" * w)
    print(f"{'*' * 4}  {text}  {'*' * 4}")
    print("*" * w)
    print()


def print_subbanner(text):
    print(f"## {text} ##")


def disable_output_stream_buffering():
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
    sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), "wb", 0), write_through=True)


class SuppressedStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Nestedspace(argparse.Namespace):
    """This is a namespace that can be used with argparse, where names
    containing a '.' are put a sub-namespaces."""
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value


def autoclean_tempdir(suffix=None, prefix=None, dir=None):
    """Returns the name of a temporary directory that will be cleaned up upon
    normal program termination. This function operates exactly as
    `tempfile.TemporaryDirectory()` does, except that the cleanup is delayed
    until program termination. The `prefix`, `suffix`, and `dir` arguments
    are the same as for `tempfile.mkdtemp()` and passed to it verbatim.
    """
    def cleanup(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
    dirname = tempfile.mkdtemp(suffix, prefix, dir)
    atexit.register(cleanup, dirname)
    return dirname


def str2bool(v):
    """A custom parser for argparse. Can be used like this:
      parser.add_argument(
          "--param", type=str2bool, nargs="?", const=True, default=False)
    And on the command line we can specify either:
      script --param
    Or:
      script --param <bool>
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected boolean value.")


def print_namespace(ns, prefix=""):
    args = vars(ns)
    for k, v in sorted(args.items()):
        if type(v) == Nestedspace:
            print_namespace(v, f"{k}.")
        if "name" in dir(v):
            # Type is, for example, NamedOptionalSet.
            print(f"{prefix}{k}: '{v.name}'")
        else:
            print(f"{prefix}{k}: {v}")

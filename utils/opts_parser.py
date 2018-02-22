"""Collect command-line options in a dictionary"""
import sys
import json

def _getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def getopts():
    from sys import argv
    args = _getopts(argv)
    assert '--config' in args, "No config file"
    config_file = args['--config']
    cfg = json.load(open(config_file))
    return cfg
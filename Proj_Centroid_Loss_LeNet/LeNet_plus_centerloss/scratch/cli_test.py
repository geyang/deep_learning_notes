#! /usr/bin/python3

import os, sys
from argparse import ArgumentParser

parser = ArgumentParser()

print(os.environ['ENV_FOO'])


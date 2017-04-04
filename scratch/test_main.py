from termcolor import cprint, colored as c
from test_module import constant

cprint(c(constant, 'red') + c(' this is green', 'green'))

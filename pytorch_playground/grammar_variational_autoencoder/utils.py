import sys
from termcolor import cprint as _cprint, colored as c
from pprint import pprint
import traceback


class Ledger():
    def __init__(self, debug=True):
        self.is_debug = debug
        pass

    def p(self, *args, **kwargs):
        self.print(*args, **kwargs)

    def print(self, *args, **kwargs):
        """use stdout.flush to allow streaming to file when used by IPython. IPython doesn't have -u option."""
        print(*args, **kwargs)
        sys.stdout.flush()

    def cp(self, *args, **kwargs):
        self.cprint(*args, **kwargs)

    def cprint(self, *args, sep=' ', color='white', **kwargs):
        """use stdout.flush to allow streaming to file when used by IPython. IPython doesn't have -u option."""
        _cprint(sep.join([str(a) for a in args]), color, **kwargs)
        sys.stdout.flush()

    def pp(self, *args, **kwargs):
        self.pprint(*args, **kwargs)

    def pprint(self, *args, **kwargs):
        pprint(*args, **kwargs)
        sys.stdout.flush()

    def log(self, *args, **kwargs):
        """use stdout.flush to allow streaming to file when used by IPython. IPython doesn't have -u option."""
        self.print(*args, **kwargs)

    # TODO: take a look at https://gist.github.com/FredLoney/5454553
    def debug(self, *args, **kwargs):
        # DONE: current call stack instead of last traceback instead of.
        if self.is_debug:
            stacks = traceback.extract_stack()
            last_caller = stacks[-2]
            path = last_caller.filename.split('/')
            self.white(path[-2], end='/')
            self.green(path[-1], end=' ')
            self.white('L', end='')
            self.red('{}:'.format(last_caller.lineno), end=' ')
            self.grey(last_caller.line)
            self.white('----------------------')
            self.print(*args, **kwargs)

    def refresh(self, *args, **kwargs):
        """allow keyword override of end='\r', so that only last print refreshes the console."""
        # to prevent from creating new line
        # default new end to single space.
        if 'end' not in kwargs:
            kwargs['end'] = ' '
        self.print('\r', *args, **kwargs)

    def info(self, *args, **kwargs):
        self.cprint(*args, color='blue', **kwargs)

    def error(self, *args, sep='', **kwargs):
        self.cprint(*args, color='red', **kwargs)

    def warn(self, *args, **kwargs):
        self.cprint(*args, color='yellow', **kwargs)

    def highlight(self, *args, **kwargs):
        self.cprint(*args, color='green', **kwargs)

    def green(self, *args, **kwargs):
        self.cprint(*args, color='green', **kwargs)

    def grey(self, *args, **kwargs):
        self.cprint(*args, color='grey', **kwargs)

    def red(self, *args, **kwargs):
        self.cprint(*args, color='red', **kwargs)

    def yellow(self, *args, **kwargs):
        self.cprint(*args, color='yellow', **kwargs)

    def blue(self, *args, **kwargs):
        self.cprint(*args, color='blue', **kwargs)

    def magenta(self, *args, **kwargs):
        self.cprint(*args, color='magenta', **kwargs)

    def cyan(self, *args, **kwargs):
        self.cprint(*args, color='cyan', **kwargs)

    def white(self, *args, **kwargs):
        self.cprint(*args, color='white', **kwargs)

    # def assert(self, statement, warning):
    #     if not statement:
    #         self.error(warning)
    #

    def raise_(self, exception, *args, **kwargs):
        self.error(*args, **kwargs)
        raise exception


class Struct():
    def __init__(self, **d):
        """Features:
        0. Take in a list of keyword arguments in constructor, and assign them as attributes
        1. Correctly handles `dir` command, so shows correct auto-completion in editors.
        2. Correctly handles `vars` command, and returns a dictionary version of self. 
        
        When recursive is set to False, 
        """
        # double underscore variables are mangled by python, so we use keyword argument dictionary instead.
        # Otherwise you will have to use __Struct_recursive = False instead.
        if '__recursive' in d:
            __recursive = d['__recursive']
            del d['__recursive']
        else:
            __recursive = True
        self.__is_recursive = __recursive
        # keep the input as a reference. Destructuring breaks this reference.
        self.__d = d

    def __dir__(self):
        return self.__dict__.keys()

    def __str__(self):
        return str(self.__dict__)

    def __getattr__(self, key):
        value = self.__d[key]
        if type(value) == type({}) and self.__is_recursive:
            return Struct(**value)
        else:
            return value

    def __getattribute__(self, key):
        if key == "_Struct__d" or key == "__dict__":
            return super().__getattribute__("__d")
        elif key in ["_Struct__is_recursive", "__is_recursive"]:
            return super().__getattribute__("__is_recursive")
        else:
            return super().__getattr__(key)

    def __setattr__(self, key, value):
        if key == "_Struct__d":
            super().__setattr__("__d", value)
        elif key == "_Struct__is_recursive":
            super().__setattr__("__is_recursive", value)
        else:
            self.__d[key] = value


def forward_tracer(self, input, output):
    _cprint(c("--> " + self.__class__.__name__, 'red') + " ===forward==> ")


def backward_tracer(self, input, output):
    _cprint(c("--> " + self.__class__.__name__, 'red') + " <==backward=== ")


ledger = Ledger()

if __name__ == "__main__":
    import time

    # print('running test as main script...')
    # ledger.log('blah_1', 'blah_2')
    # for i in range(10):
    #     ledger.refresh('{}: hahahaha'.format(i))
    #     ledger.green('hahaha', end=" ")
    #     time.sleep(0.5)

    # test dictionary to object
    test_dict = {
        'a': 0,
        'b': 1
    }

    test_args = Struct(**test_dict)
    assert test_args.a == 0
    assert test_args.b == 1
    test_args.haha = 0
    assert test_args.haha == 0
    test_args.haha = {'a': 1}
    assert test_args.haha != {'a': 1}
    assert vars(test_args.haha) == {'a': 1}
    assert test_args.haha.a == 1
    assert test_args.__dict__['haha']['a'] == 1
    assert vars(test_args)['haha']['a'] == 1
    print(test_args)

    test_args = Struct(__recursive=False, **test_dict)
    assert test_args.__is_recursive == False
    assert test_args.a == 0
    assert test_args.b == 1
    test_args.haha = {'a': 1}
    assert test_args.haha['a'] == 1
    assert test_args.haha == {'a': 1}

    ledger.green('*Struct* tests have passed.')

    # Some other usage patterns
    test_args = Struct(**test_dict, **{'ha': 'ha', 'no': 'no'})
    print(test_args.ha)

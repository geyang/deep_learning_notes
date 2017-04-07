import sys
from termcolor import cprint as _cprint, colored as c
from pprint import pprint


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
    # def raise(self, exception):
    #     raise exception


class Struct():
    def __init__(self, **d):
        # keep the input as a reference. Destructuring breaks this reference.
        self.__dict__['d'] = d

    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return Struct(**value)

        return value


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
    assert Struct(**test_dict).a == 0
    assert Struct(**test_dict).b == 1
    ledger.green('obj test has passed.')

from visdom import Visdom

class Dashboard(Visdom):
    def __init__(self, name):
        super(Dashboard, self).__init__()
        self.env = name
        self.plots = {}
        self.plot_data = {}

    def plot(self, name, type, *args, **kwargs):
        if 'opts' not in kwargs:
            kwargs['opts'] = {}
        if 'title' not in kwargs['opts']:
            kwargs['opts']['title'] = name

        if hasattr(self, type):
            if name in self.plots:
                getattr(self, type)(win=self.plots[name], *args, **kwargs)
            else:
                id = getattr(self, type)(*args, **kwargs)
                self.plots[name] = id
        else:
            raise AttributeError('plot type: {} does not exist. Please'
                                 'refer to visdom documentation.'.format(type))

    def append(self, name, type, *args, **kwargs):
        if name in self.plots:
            self.plot(name, type, *args, update='append', **kwargs)
        else:
            self.plot(name, type, *args, **kwargs)

    def remove(self, name):
        del self.plots[name]

    def clear(self):
        self.plots = {}

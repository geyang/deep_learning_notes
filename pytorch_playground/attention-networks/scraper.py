import sys
from bs4 import BeautifulSoup
import requests
import re

print('******* {} Module is reloaded *******'.format(__name__))

class Page():
    def __init__(self, url, debug=False):
        self.url = url
        self._debug = debug

    def request(self):
        page = requests.get(self.url)
        self.soup = BeautifulSoup(page.content, 'html.parser')
        return self

    def set_mask(self, regexString):
        self.url_mask = re.compile(regexString)
        return self

    def test_mask(self, string):
        return self.url_mask.match(string)

    def get_anchors(self):
        for n in self.soup.findAll('a'):
            if 'href' in n.attrs and self.url_mask.match(n['href']) is not None:
                yield self.url_mask.match(n['href']), n

    def debug(self, output, *args, **kwargs):
        if self._debug:
            print(output, *args, **kwargs)
            sys.stdout.flush()

    def download(self, path, mode='wb', chunk_size=None):
        r = requests.get(self.url, stream=True)
        with open(path, mode) as f:
            self.debug('Request to {} has been made'.format(self.url), end='')
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    self.debug('.'.format(self.url), end='')
            self.debug('. '.format(self.url), end='')
        self.debug('File {} has been saved.'.format(path))

from scraper import Page

p = Page('http://www.manythings.org/anki/', debug=True)
p.set_mask('(.*).zip')
p.request()

for m, n in p.get_anchors():
    n_p = Page(p.url + m[0], debug=True)
    n_p.download('./data/' + m[0], 'wb', chunk_size=4096*2**4)
import codecs

import pycaption

folder = '/media/graham/OS/Linux Content/Youtube/'
filename = folder+'GD7GNO08Epg.mp4.en.vtt'

try:
	captions = codecs.open(filename, 'r').read()
except:
	captions = open(filename,'r').read()
	captions = unicode(captions, errors='replace')

vttReader = pycaption.WebVTTReader()
lang = 'en-US'

if vttReader.detect(captions):
	print('It looks like a vtt file!')
	C = vttReader.read(captions,lang)

try:
	caps = C.get_captions(lang)
	print('It smells like a vtt file!')
except:
	print('Error!')

for n in range(0,100):
	print('%s:\t%s' % (caps[n].format_start(), caps[n].get_text()))
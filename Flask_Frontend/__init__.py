from flask import Flask
import matplotlib
matplotlib.use('Agg')
# Force matplotlib to not use any Xwindows backend
# 	needs to run before any other import of matplotlib, so 
#	resides here in the Flask __init__ file
#
# 	(this is necessary to run matplotlib over a ssh connection)
# 	(I also tried running ssh with the -X flag, but this gave me errors
#			when I tried to logout of the ssh session)


app = Flask(__name__)
from Flask_Frontend import views
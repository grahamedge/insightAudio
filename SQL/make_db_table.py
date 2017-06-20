import models
from models import engine

# This next line creates the table if it doesn't exists already.
# The database specified in setting.py must already exist before you run this.
models.AudioInfo.metadata.create_all(engine)

models.VideoSummary.metadata.create_all(engine)

from datathon.db import Database
from datathon.preprocessing.clean import load_and_clean

db = Database('data/raw/datathon.db')
df = load_and_clean(db)
print(df.count())

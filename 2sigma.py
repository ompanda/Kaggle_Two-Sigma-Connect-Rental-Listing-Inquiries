from subprocess import check_output
import pandas as pd


print(check_output(["ls", "./data"]).decode("utf8"))

df_test = pd.read_json(open("./data/test.json", "r"))

#print(df_test["listing_id"])

print(df_test["listing_id"].count())
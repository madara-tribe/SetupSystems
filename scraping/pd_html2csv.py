# pip3 install lxml
import pandas as pd

html = "https://www.***.or.jp/***"
csv = "****.csv"
tables = pd.read_html(html)
tables[0].to_csv(csv)

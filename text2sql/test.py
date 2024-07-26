from text2sql.core import Text2SQL

sql = Text2SQL(model = "gpt-3.5-turbo")
query = sql.query("How much do we have in total sales?")
print(query)




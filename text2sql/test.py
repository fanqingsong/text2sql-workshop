

from dotenv import load_dotenv
import os, time

load_dotenv()

print("--------------------------")
print(os.environ['OPENAI_API_KEY'])


from text2sql.core import Text2SQL



sql = Text2SQL()
# query = sql.query("How much do we have in total sales?")
# print("-------------------------")
# print(query)

# result = sql.run_sql(query)
# print(result)


result = sql.run("How much do we have in total sales?")
# result = sql.run("how many items are there in sales table?")
# result = sql.run("Describe the sales table")
print(result)



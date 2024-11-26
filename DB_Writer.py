import pandas as pd
import psycopg2

data_frame = pd.read_csv("C:\\Users\\AnthE\\Desktop\\Personal Projects\\IT_CaseComp_2024\\Dummy Data\\Ecommerce Order Dataset\\train\\df_Products.csv")

conn = psycopg2.connect(
    dbname="CentralRepo",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

print(data_frame.columns)

table_name = "Products"
columns = []
for column in data_frame.columns:
    columns.append(f"{column} VARCHAR(240)")

create_table = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"

cursor.execute(create_table)

for index,row in data_frame.iterrows():
    insert_rows = f"INSERT INTO {table_name} ({', '.join(data_frame.columns)}) VALUES ({', '.join(['%s'] * len(data_frame.columns))})"
    cursor.execute(insert_rows, tuple(row))

conn.commit()

cursor.close()
conn.close()
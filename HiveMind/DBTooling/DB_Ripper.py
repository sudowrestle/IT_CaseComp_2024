import pandas as pd
import psycopg2

conn = psycopg2.connect(
    dbname="CentralRepo",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

table_name = "orders"

select_query = f"SELECT * from orders WHERE order_delivered_timestamp != 'NaN' AND order_delivered_timestamp >= '2018' ORDER by order_delivered_timestamp DESC LIMIT 100"
data_frame = pd.read_sql(select_query, conn)

data_frame.to_csv('./Dummy Data/DBRIPS/yo.csv', index=False)

conn.close()
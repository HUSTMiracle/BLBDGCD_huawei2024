import psycopg2
import os
import json
class PostgresModel:
    def __init__(self):
        # migration
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS history (image_id VARCHAR(255) PRIMARY KEY, json_data TEXT)")
        conn.commit()
        cur.close()
        conn.close()
    def connect(self):
        conn = psycopg2.connect(
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"]
        )
        return conn
    def get_all_histories(self,page,limit,date_string):
        conn = self.connect()
        cur = conn.cursor()
        if page is None:
            cur.execute("SELECT image_id,json_data FROM history WHERE image_id LIKE %s ORDER BY image_id DESC", (date_string + '%'))
        else:
            cur.execute("SELECT image_id,json_data FROM history WHERE image_id LIKE %s ORDER BY image_id DESC LIMIT %s OFFSET %s", (date_string + '%', limit, (int(page)-1)*int(limit)))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    def insert_inference_result(self,image_id,inference_json):
        conn = self.connect()
        cur = conn.cursor()
        dumped_json = json.dumps(inference_json)
        cur.execute("INSERT INTO history (image_id,json_data) VALUES (%s,%s)", (image_id,dumped_json))
        conn.commit()
        cur.close()
        conn.close()
    def get_inference_json(self,image_id):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT json_data FROM history WHERE image_id = %s", (image_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return json.loads(row[0])

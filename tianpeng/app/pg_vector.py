import psycopg2


def pg_conn():
    return psycopg2.connect(
        dbname="default_database",  # Replace with your database name
        user="admin",  # Replace with your username
        password="admin123",  # Replace with your password
        host="localhost",  # Replace with your host
        port="5432",  # Replace with your port
    )


def create_extension(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def test_init_curs(cur, conn):
    # Create a table
    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS example_table (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100),
          age INTEGER
      )
  """
    )
    conn.commit()
    print("Table created successfully")

    # Insert a record into the table
    cur.execute("INSERT INTO example_table (name, age) VALUES (%s, %s)", ("John", 30))
    conn.commit()
    print("Record inserted successfully")


def init():
    # Establish connection to the PostgreSQL database
    conn = pg_conn()

    # Create a cursor object to interact with the database
    cur = conn.cursor()

    # create extension
    create_extension(cur)

    # Close cursor and connection
    cur.close()
    conn.close()

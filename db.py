import psycopg2


def get_connection():

    connection = psycopg2.connect(
        host="localhost",
        port=5432,
        database="birza",
        user="postgres",
        password="postgres"
    )

    return connection


if __name__ == "__main__":

    connection = get_connection()

    print("Подключение к PostgreSQL успешно")

    connection.close()
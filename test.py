from sensor.configuration.mongo_db_connection import MongoDBClient


if __name__ == '__main__':

    mongodb = MongoDBClient()
    print("collection_name:", mongodb.database.list_collection_names())

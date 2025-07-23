from pymongo import MongoClient

def get_mongo_collection(mongo_uri, db_name, collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]

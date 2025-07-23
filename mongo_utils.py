# mongo_utils.py

from pymongo import MongoClient

def get_mongo_client(uri: str) -> MongoClient:
    return MongoClient(uri)

def get_mongo_collection(uri: str, db_name: str, collection_name: str):
    client = get_mongo_client(uri)
    db = client[db_name]
    return db[collection_name]

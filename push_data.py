import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logger
 
# Load environment variables from the .env file
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def csv_to_json_convertor(self, csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def push_data_to_mongodb(self, records, database, collection):
        try :
            self.records = records
            self.database = database
            self.collection = collection
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)

            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    File_Path = "/Users/tranminhanh/Downloads/Data Science/NetworkSecurity/network_data/phisingData.csv"
    DATABASE = "NetworkData"
    Collection = "PhishingData"
    network_object = NetworkDataExtract()
    records = network_object.csv_to_json_convertor(File_Path)
    number_of_records = network_object.push_data_to_mongodb(records, DATABASE, Collection)
    print(f"Number of records inserted: {number_of_records}")
   
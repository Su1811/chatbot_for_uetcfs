import os
import pprint
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DB_URI")
client = MongoClient(DB_URI)


def add_content(content, topic, sentiment, status):
    """
    Inserts a content into the content collection, with the following fields:

    - "content"
    - "topic"
    - "sentiment"
    - "status"

    """

    cnt = {
        'content': content,
        'topic': topic,
        'sentiment': sentiment,
        'status': status
    }

    inserted_id = client.test.contents.insert_one(cnt)
    print(inserted_id)


printer = pprint.PrettyPrinter()


def get_all():
    contents = client.test.contents.find()
    for content in contents:
        printer.pprint(content)


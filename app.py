
import configparser
from flask import Flask, render_template, request, Response
from flask.json import JSONEncoder
import requests, json, random, os
from data import data_preprocessing
from db import *
from utils import *
from dotenv import load_dotenv
from timeit import default_timer as timer


load_dotenv()
app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
DB_URI = os.getenv('DB_URI')

model_topic = get_model(1)
model_sentiment = get_model(2)
model_status = get_model(3)


@app.route('/webhook', methods=['GET'])
def webhook_verify():
    if request.args.get('hub.mode') == "subscribe" and request.args.get('hub.verify_token') == VERIFY_TOKEN:
        return Response(response=request.args.get('hub.challenge'), status=200)
    return Response(response="Wrong verify token", status=403)


@app.route('/webhook', methods=['POST'])
# Accepts the POST request and saves the content of the event
def webhook_action():
    data = json.loads(request.data.decode('utf-8'))
    # Checks if this is an event from a page subscription
    if data['object'] == "page":

        for entry in data['entry']:
            if 'message' in entry['messaging'][0]:
                user_message = entry['messaging'][0]['message']['text']
                user_id = entry['messaging'][0]['sender']['id']
                start = timer()
                handle_message(user_id, user_message)
                end = timer()
                print('{:.2f}'.format(end - start))

        # Returns a '200 OK' response to all requests
        return Response(response="EVENT RECEIVED", status=200)
    else:
        # Returns a '404 Not Found' if event is not from a page subscription
        return Response(response="404 Not Found", status=404)


def handle_message(user_id, user_message):
    data_input_ids, data_attention_masks = data_preprocessing(user_message)
    topic, _ = predict(model_topic, data_input_ids, data_attention_masks, device)
    if topic < 3:
        sentiment, _ = predict(model_sentiment, data_input_ids, data_attention_masks, device)
        _, status = predict(model_status, data_input_ids, data_attention_masks, device)
        status = '{:.2f}'.format(status * 100)
    else:
        sentiment = 1
        status = 0

    # Topic
    if topic == 0:
        topic = "Học tập"
    elif topic == 1:
        topic = "Đời sống"
    elif topic == 2:
        topic = "Others"
    else:
        topic = "Spam"
    # Sentiment
    if sentiment == 0:
        sentiment = "Tiêu cực"
    elif sentiment == 1:
        sentiment = "Trung lập"
    else:
        sentiment = "Tích cực"

    add_content(user_message, topic, sentiment, status)
    get_all()

    # Status
    status = f"{status}% được đăng"

    r_content = str(f'Cảm ơn bạn đã chia sẻ thông tin! \nDebug message … \nKết quả dự đoán: \n\tChủ đề: {topic} \n\t'
                    f'Cảm xúc: {sentiment} \n\tTrạng thái: {status}')
    print(type(r_content))
    response = {
        'recipient': {'id': user_id},
        'message': {'text': r_content},
    }


    r = requests.post('https://graph.facebook.com/v16.0/me/messages/?access_token=' + PAGE_ACCESS_TOKEN, json=response)
    print(r.content)

# class MongoJsonEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, datetime):
#             return obj.strftime("%Y-%m-%d %H:%M:%S")
#         if isinstance(obj, ObjectId):
#             return str(obj)
#         return json_util.default(obj, json_util.CANONICAL_JSON_OPTIONS)
#

if __name__ == "__main__":
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)

    if os.path.isfile("./bert_models/bertLSTM_category.pt"):
        load_model(model_topic, "./bert_models/bertLSTM_category.pt", device)
    if os.path.isfile("./bert_models/bertbase_sentiment.pt"):
        load_model(model_sentiment, "./bert_models/bertbase_sentiment.pt", device)
    if os.path.isfile("./bert_models/bertLSTM_status.pt"):
        load_model(model_status, "./bert_models/bertLSTM_status.pt", device)

    # app.json_encoder = MongoJsonEncoder
    # app.config['DEBUG'] = True
    app.config['MONGO_URI'] = DB_URI

    app.run()

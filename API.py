import json, string, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pyvi import ViTokenizer, ViPosTagger
from bs4 import BeautifulSoup
import pickle
from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS,cross_origin

def tienxuly_vanban(text):
    soup = BeautifulSoup(text)
    text = " ".join(soup.get_text().split("\n"))
    text = ViTokenizer.tokenize(text)
    text = text.lower()
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"[0-9]+", " num ", text)
    for punc in string.punctuation:
        if punc != "_":
            text = text.replace(punc, ' ')
    text = re.sub('\\s+',' ',text)
    return text

def read_json_file(json_data):
    id_list = []
    content_list = []
    title_list = []
    photo_list = []
    for p in json_data:
        if p['content'] != None:
            id_list.append(p['id'])
            content_list.append(tienxuly_vanban(p['content']))
            title_list.append(p['title'])
            photo_list.append(p['photo'])
    df = pd.DataFrame(list(zip(id_list, title_list, content_list,photo_list)), columns =['ID', 'Title', "Content", "Photo"])
    with open('database.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df


def calculate_tfidf_matrix(df_data):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
    tfidf_matrix = tf.fit_transform(df_data['Content'])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    model = {}

    for idx, row in df_data.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_item = [(cosine_similarities[idx][i], df_data['ID'][i]) for i in similar_indices]
        model[row['ID']] = similar_item[1:]
    return model

def train(json_data):
    df_data = read_json_file(json_data)
    model = calculate_tfidf_matrix(df_data)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Done")

# Hàm dự đoán. đầu vào là id của bài báo, đầu ra số lượng bài tương đồng với bài báo.
def predict(paper_id, threhold = 4):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    outputs = model[paper_id][:threhold]
    output_json = []
    with open('database.pickle', 'rb') as handle:
        database = pickle.load(handle)
    for i, output in enumerate(outputs):
        x = {'ID':str(output[1]),
             'Similarity':str(output[0]),
             "Title": str(database.loc[database['ID'] == int(output[1]), 'Title'].iloc[0]),
             "Photo": str(database.loc[database['ID'] == int(output[1]), 'Photo'].iloc[0])
             }
        output_json.append(x)
    app_json = json.dumps(output_json, ensure_ascii=False)
    return app_json

app = Flask('__name__')
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'GET':
        return 'Hello world'
    elif request.method == 'POST':
        headers = request.headers
        action = headers.get("action")
        if action == 'train':
            content = request.json
            train(content)
            return jsonify({"message": "TRAIN MODEL DONE"}), 200
        elif action == 'predict':
            print(request)
            content = request.json
            print(request)
            print(content)
            id = content["ID"]
            print("ID: ", id)
            print(request)
            dict_output = predict(id, threhold = 4)
            print(request)
            return jsonify(dict_output.encode().decode('utf-8')), 200
        else:
            return jsonify({"message": "ERROR"}), 401


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=33)
    app.run()

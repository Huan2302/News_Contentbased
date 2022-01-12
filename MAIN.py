import json,string, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pyvi import ViTokenizer, ViPosTagger

from bs4 import BeautifulSoup

def tienxuly_vanban(text):
    soup = BeautifulSoup(text)
    text = " ".join(soup.get_text().split("\n"))
    text = ViTokenizer.tokenize(text)
    text = text.lower()
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"[0-9]+", " num ", text)
    for punc in string.punctuation:
      if punc !="_":
          text = text.replace(punc,' ')
    text = re.sub('\\s+',' ',text)
    return text

def read_json_file(path_data):
  id_list = []
  content_list = []
  title_list = []
  with open(path_data) as json_file:
    data = json.load(json_file)
    for p in data:
      if p['content'] != None:
        id_list.append(p['id'])
        content_list.append(tienxuly_vanban(p['content']))
        title_list.append(p['title'])
  df = pd.DataFrame(list(zip(id_list, title_list, content_list)), columns =['ID', 'Title', "Content"])
  return df

df_data = read_json_file("data.json")
df_data.head()

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
tfidf_matrix = tf.fit_transform(df_data['Content'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in df_data.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], df_data['ID'][i]) for i in similar_indices]
    results[row['ID']] = similar_items[1:]

print('done!')

def return_title(id):
    return df_data.loc[df_data['ID'] == id]['Title'].tolist()[0]
def return_content(id):
    return df_data.loc[df_data['ID'] == id]['Content'].tolist()[0]

def recommend(item_id, num):
    print("=" * 100)
    print("Input")
    print("ID: ", item_id)
    print("Title: ", return_title(item_id))
    print("Content: ", return_content(item_id))
    print("=" * 100)
    print("=" * 100)

    print("Output")
    print("-" * 50)
    recs = results[item_id][:num]
    for i, rec in enumerate(recs):
        print("STT: ", i)
        print("Similarity: ", rec[0])
        print("ID: ", rec[1])
        print("Title: ", return_title(rec[1]))
        print("Content: ", return_content(rec[1]))
        print("*" * 50)
    print("-" * 50)
recommend(item_id=1, num=4)
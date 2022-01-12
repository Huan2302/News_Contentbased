import requests
import json

# Đánh giá trên bộ dữ liệu thu thập


# Đọc dữ liệu file data dưới dạng json.
with open("data.json") as json_file:
    data_json = json.load(json_file)

key = {"action":"train"}
r = requests.get('http://192.168.1.5:33/', json=data_json,headers=key)
print(r.content)
import json
from fastapi import FastAPI


app = FastAPI()
['title', 'price', 'description', 'category', 'images']


@app.get("/products")
def json_product():
    with open('app/extracted_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for collection in data["storeData"]["products"]:
        del collection['id']
        del collection['createdAt']
        del collection['updatedAt']

    for images in data["storeData"]["products"][0]["images"]:
        del images['id']
        del images['url']

    # for images in data["storeData"]["products"][0]["images"]:
    #     del images['id']
    #     del images['url']

    # print(data["storeData"]["products"])
    print(type(data))

    return {"products": data["storeData"]["products"]}

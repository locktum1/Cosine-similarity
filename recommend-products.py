from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json

app = Flask(__name__)

def similarProduct(products_data, product_name):
    df = pd.DataFrame(products_data)

    similarity = cosine_similarity(df.T)

    product_names = df.columns.tolist()
    
    if product_name not in product_names:
        return None
        
    product_index = product_names.index(product_name)
    
    similarities = similarity[product_index]
    
    similarities[product_index] = -1
    
    most_similar_index = np.argmax(similarities)
    
    most_similar_product = product_names[most_similar_index]
    
    return most_similar_product

@app.route('/recommend',method=['POST'])
def recommend():
    data = request.get_json()
    
    product_name = data['product_name']
    products_data = data['products_data']
    
    most_similar_product = similarProduct(products_data, product_name)
    
    if not most_similar_product:
        return jsonify({'error': f'Not found'})
    
    
    return jsonify({most_similar_product})

if __name__ == '__main__':
    main()















#
#app = Flask(__name__)
#
##Fake product data for testing purposes
#products = pandas.DataFrame({
#'productA': [1, 0, 1, 0.2],
#'productB': [0, 0, 1, 0.5],
#'productC': [1, 1, 0, 0],
#'productD': [1, 0, 1, 0.9],
#'productE': [1, 0, 1, 0.5]
#}, index=['electronics','clothes','expensive','mid-range']).T
#
#def find_similarity(bought_products):
#    similarities = {}
#    for bought_product in bought_products:
#        if bought_product not in products.index:
#            continue
#        bought_vector = products.loc[bought_product].values.reshape(1,-1)
#        
#        for product in products.index:
#            if product == bought_product:
#                continue
#                
#            bought_vector = products.loc[bought_product].values.reshape(1,-1)
#            
#            if np.linalg.norm(bought_vector) == 0 or np.linalg.norm(product_vector) == 0:
#                similarity = 0
#            else:
#                similarity = 1 - cosine(bought_vector, product_vector)
#            
#            similarities[product] = similarities.get(product, 0) + similarity
#            
#    recommended = sorted(similarities, key=similarities.get, reverse=True)[:3]
#    return recommended
#
#@app.route('/recommend', methods=['GET'])
#def recommend():
#    bought_products = request.args.getlist('products')
#        
#    if not bought_products:
#        return jsonify({'error': 'No products provided'}), 400
#        
#    recommendations = find_similarity(bought_products)
#    return jsonify({'recommended_products': recommendations})
#    
#if __name__ == '__main__':
#    app.run(host='0.0.0.0',port=5000)

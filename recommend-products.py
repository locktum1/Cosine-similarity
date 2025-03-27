from flask import Flask, request, jsonify
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def main():
    try:
        data = request.json
        products = data.get('products', [])
        boughtProducts = data.get('boughtProducts', [])

        results = []

        for product in boughtProducts:
            df = pd.DataFrame(products)

            product_df = pd.DataFrame([product])

            encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
            encoded = encoder.fit_transform(df[['brand','category']])

            product_encoded = encoder.transform(product_df[['brand','category']])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[['price']].astype(float))

            product_scaled = scaler.transform(product_df[['price']].astype(float))

            features = np.hstack((encoded,scaled))

            product_features = np.hstack((product_encoded, product_scaled))

            matrix = cosine_similarity(features, product_features)

            most_similar_index = np.argmax(matrix)

            most_similar_product = df.iloc[most_similar_index]

            resultID = most_similar_product['product_id']

            results.append({
                'most_similar_product': most_similar_product.to_dict(),
                'cosine_similarity': matrix[most_similar_index][0]
            })

            del products[most_similar_index]

        return jsonify({'result': results})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder='templates')

# Load trained model
model = joblib.load('ml_chatbot_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba([user_input])[0]
            max_proba = max(probas)
            predicted_category = model.classes_[probas.argmax()]

            if max_proba < 0.3:
                response = "I'm not sure what you mean. Can you try rephrasing?"
            else:
                response = get_response(predicted_category)
        else:
            predicted_category = model.predict([user_input])[0]
            response = get_response(predicted_category)
    except Exception as e:
        print("Prediction error:", e)
        response = "Something went wrong."

    return jsonify({"response": response})

def get_response(category):
    responses = {
        "intro": "Machine Learning is a field of AI that enables computers to learn from data.",
        "algorithms": "Some common algorithms include Linear Regression, Decision Trees, and SVM.",
        "supervised": "Supervised Learning involves training a model on labeled data.",
        "unsupervised": "Unsupervised Learning is used when the data is unlabeled, like clustering.",
        "eval": "Model evaluation includes metrics like accuracy, precision, recall, and F1-score.",
        "course": "This course covers ML basics, algorithms, supervised/unsupervised learning, and model evaluation."
    }
    return responses.get(category, "Sorry, I don't understand that question.")

if __name__ == '__main__':
    app.run(debug=True)

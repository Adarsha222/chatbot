from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Example training data
X = [
     "What is machine learning?",
    "Define machine learning",
    "Explain machine learning",
    "Tell me about machine learning",
    "ML basics",
    "Can you explain ML?",
    "Intro to machine learning",
    
    "What is supervised learning?",
    "Explain supervised learning",
    "Supervised vs unsupervised?",
    "Tell me about labeled data",

    "What is unsupervised learning?",
    "Explain clustering",
    "What is K-means?",
    
    "List some ML algorithms",
    "Which ML algorithms are popular?",
    "What are common machine learning models?",

    "How do you evaluate a model?",
    "What is precision and recall?",
    "Explain model testing",

    "What will I learn in the course?",
    "Course overview for ML",
    "What topics are covered in this course?"
]

y = [
    "intro",
    "intro",
    "intro",
    "intro",
    "intro",
    "intro",
    "intro",
    "supervised",
    "supervised",
    "supervised",
    "supervised",
    "unsupervised",
    "unsupervised",
    "unsupervised",
    "algorithms",
    "algorithms",
    "algorithms",
    "eval",
    "eval",
    "eval",
    "course",
    "course",
    "course"
]

# Train the model
model = make_pipeline(TfidfVectorizer(lowercase=True), MultinomialNB())
model.fit(X, y)

# Save model to file
joblib.dump(model, 'ml_chatbot_model.pkl')
print("✅ Model trained and saved as ml_chatbot_model.pkl")

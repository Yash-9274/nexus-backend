import nltk
import os

def download_nltk_data():
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.dirname(__file__), '../../nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'stopwords'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

# Call this function when your app starts 
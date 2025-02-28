from typing import BinaryIO, Dict
import PyPDF2
from docx import Document as DocxDocument
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import openai
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data at startup
import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('words')

download_nltk_data()

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english'
        )

    def process_pdf(self, file: BinaryIO) -> str:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def process_docx(self, file: BinaryIO) -> str:
        doc = DocxDocument(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def process_markdown(self, content: str) -> str:
        return markdown.markdown(content)

    def create_embeddings(self, text: str) -> list[float]:
        """Create document embeddings using TF-IDF as fallback"""
        try:
            # Fit and transform the text
            vector = self.vectorizer.fit_transform([text])
            # Convert sparse matrix to dense and normalize
            dense_vector = vector.todense()
            normalized = dense_vector / np.linalg.norm(dense_vector)
            # Convert to list and ensure consistent size
            embedding = normalized.tolist()[0]
            # Pad or truncate to exactly 512 dimensions
            if len(embedding) < 512:
                embedding.extend([0] * (512 - len(embedding)))
            return embedding[:512]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return [0] * 512  # Return zero vector as fallback

    async def analyze_document(self, text: str) -> Dict:
        try:
            # Check if OpenAI API key is configured
            if not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured, falling back to basic analysis")
                return self.basic_analysis(text)

            # Split text into chunks for longer documents
            chunks = self.text_splitter.split_text(text)
            first_chunk = chunks[0] if chunks else text[:4000]
            
            try:
                # Get AI analysis from OpenAI
                response = await openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "You are an expert document analyzer. Extract key information in a structured JSON format."
                    }, {
                        "role": "user",
                        "content": f"""Analyze this text and provide the following in JSON format:
                        1) summary: A concise 2-3 sentence summary
                        2) keywords: Array of 5-10 most important keywords
                        3) entities: Array of named entities (people, organizations, locations)
                        4) category: The document category/type

                        Text: {first_chunk}"""
                    }]
                )
                
                ai_analysis = json.loads(response.choices[0].message.content)
                embedding = self.create_embeddings(text)

                return {
                    "summary": ai_analysis.get("summary", ""),
                    "keywords": ai_analysis.get("keywords", []),
                    "entities": ai_analysis.get("entities", []),
                    "category": ai_analysis.get("category", "Unknown"),
                    "embedding": embedding
                }

            except (openai.error.AuthenticationError, openai.error.RateLimitError) as e:
                logger.warning(f"OpenAI API error: {str(e)}, falling back to basic analysis")
                return self.basic_analysis(text)
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return self.basic_analysis(text)

    def extract_keywords(self, text: str) -> list[str]:
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Keep only nouns and adjectives
            important_words = [word for word, tag in tagged 
                             if tag.startswith(('NN', 'JJ')) 
                             and word not in self.stop_words
                             and len(word) > 2]
            
            # Get frequency distribution
            freq_dist = Counter(important_words)
            
            # Return top 10 keywords
            return [word for word, _ in freq_dist.most_common(10)]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def basic_analysis(self, text: str) -> Dict:
        """Fallback analysis when OpenAI is unavailable"""
        keywords = self.extract_keywords(text)
        
        return {
            "summary": text[:200] + "...",  # Basic summary
            "keywords": keywords,
            "entities": self.extract_entities(text),
            "category": "Document",
            "embedding": None  # Skip embedding when OpenAI is unavailable
        }

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities using NLTK"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        named_entities = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                named_entities.append(' '.join(c[0] for c in chunk))
        
        return list(set(named_entities))[:10]  # Return up to 10 unique entities 
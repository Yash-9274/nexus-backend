from typing import BinaryIO, Dict
import PyPDF2
from docx import Document as DocxDocument
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import cohere
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data at startup
import nltk

# Initialize Cohere client
co = cohere.Client(api_key=settings.COHERE_API_KEY)

logger = logging.getLogger(__name__)

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('words')

download_nltk_data()

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
        
        # Import templates
        from app.templates.academic_template import ACADEMIC_TEMPLATE
        from app.templates.business_template import BUSINESS_TEMPLATE
        # from app.templates.technical_template import TECHNICAL_TEMPLATE
        # from app.templates.letter_template import LETTER_TEMPLATE
        
        # Initialize templates dictionary
        self.templates = {
            'academic': ACADEMIC_TEMPLATE,
            'business': BUSINESS_TEMPLATE,
            # 'technical': TECHNICAL_TEMPLATE,
            # 'letter': LETTER_TEMPLATE
        }

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
            chunks = self.text_splitter.split_text(text)
            first_chunk = chunks[0] if chunks else text[:4000]
            
            try:
                response = co.generate(
                    model='command',
                    prompt=f"""Analyze this text and provide the following in JSON format:
                    1) summary: A concise 2-3 sentence summary
                    2) keywords: Array of 5-10 most important keywords
                    3) entities: Array of named entities (people, organizations, locations)
                    4) category: The document category/type

                    Text: {first_chunk}

                    Output in JSON format only.""",
                    max_tokens=500,
                    temperature=0.3,
                )
                
                ai_analysis = json.loads(response.generations[0].text)
                embedding = self.create_embeddings(text)

                return {
                    "summary": ai_analysis.get("summary", ""),
                    "keywords": ai_analysis.get("keywords", []),
                    "entities": ai_analysis.get("entities", []),
                    "category": ai_analysis.get("category", "Unknown"),
                    "embedding": embedding
                }

            except Exception as e:
                logger.error(f"Cohere API error: {str(e)}")
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

    async def apply_template(self, content: str, template_id: str) -> str:
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        try:
            # Split content into smaller chunks
            chunks = self.text_splitter.split_text(content)
            first_chunk = chunks[0] if chunks else content[:2000]

            prompt = f"""You are a document formatter. Format this document into the following sections: {', '.join(template['structure'])}

            Rules:
            1. Extract and organize content into appropriate sections
            2. Keep the original content's meaning
            3. Ensure each section has relevant content
            4. Use exactly this format for each section:
            SECTION: [section name]
            CONTENT:
            [section content]
            END

            Document to format:
            {first_chunk}
            """

            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,
            )

            # Parse sections
            sections = {}
            current_section = None
            current_content = []
            
            for line in response.generations[0].text.split('\n'):
                line = line.strip()
                if line.startswith('SECTION:'):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = line.replace('SECTION:', '').strip().lower()
                    current_content = []
                elif line.startswith('CONTENT:'):
                    continue
                elif line == 'END':
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = None
                    current_content = []
                elif current_section and line:
                    current_content.append(line)

            # Add remaining content to sections
            remaining_text = ' '.join(chunks[1:]) if len(chunks) > 1 else ''
            
            # Ensure all template sections exist
            for section in template['structure']:
                if section not in sections:
                    if remaining_text:
                        section_text = remaining_text[:1000]
                        remaining_text = remaining_text[1000:]
                        sections[section] = section_text.strip()
                    else:
                        sections[section] = f"[{section} content will be added here]"

            # Format according to template
            formatted_content = template['format'].format(**sections)
            
            # Clean up any extra whitespace
            formatted_content = '\n'.join(line.strip() for line in formatted_content.splitlines() if line.strip())
            
            return formatted_content

        except Exception as e:
            logger.error(f"Error in template application: {str(e)}")
            return content  # Return original content on error 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from umap import UMAP
import numpy as np

class DocumentClusteringService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.umap = UMAP(n_components=3, random_state=42)
        self.clustering = DBSCAN(eps=0.3, min_samples=2)
    
    def cluster_documents(self, documents):
        # Create document vectors
        vectors = self.vectorizer.fit_transform([doc.content for doc in documents])
        
        # Reduce dimensionality for visualization
        embeddings_3d = self.umap.fit_transform(vectors.toarray())
        
        # Perform clustering
        clusters = self.clustering.fit_predict(embeddings_3d)
        
        return {
            'embeddings': embeddings_3d.tolist(),
            'clusters': clusters.tolist(),
            'feature_names': self.vectorizer.get_feature_names_out().tolist()
        } 
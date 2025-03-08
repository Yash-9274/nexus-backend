from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.document import Document
from app.api.auth import get_current_user
from app.models.user import User
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/knowledge-graph", response_model=dict)
async def get_knowledge_graph(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        documents = db.query(Document).filter(Document.user_id == current_user.id).all()
        
        nodes = []
        links = []
        seen_keywords = set()
        
        # Create graph data
        for doc in documents:
            doc_id = str(doc.id)
            nodes.append({
                "id": doc_id,
                "name": doc.title,
                "category": "document",
                "val": 1,
                "color": "#ff4444"
            })
            
            # Add keyword nodes and links
            if doc.metadata_col and "keywords" in doc.metadata_col:
                for keyword in doc.metadata_col["keywords"]:
                    keyword_id = f"keyword-{keyword}"
                    if keyword_id not in seen_keywords:
                        nodes.append({
                            "id": keyword_id,
                            "name": keyword,
                            "category": "keyword",
                            "val": 0.5,
                            "color": "#4444ff"
                        })
                        seen_keywords.add(keyword_id)
                    
                    links.append({
                        "source": doc_id,
                        "target": keyword_id,
                        "strength": 1
                    })
        
        return {
            "nodes": nodes,
            "links": links
        }
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating knowledge graph"
        )

@router.get("/document-insights", response_model=dict)
async def get_document_insights(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        documents = db.query(Document).filter(Document.user_id == current_user.id).all()
        
        # Initialize data structures
        topics = defaultdict(int)
        activity = defaultdict(int)
        entities = defaultdict(int)
        embeddings = []
        categories = defaultdict(list)
        domain_keywords = defaultdict(set)
        
        # Process documents
        for doc in documents:
            if doc.metadata_col:
                # Process topics/keywords
                if "keywords" in doc.metadata_col:
                    for keyword in doc.metadata_col["keywords"]:
                        topics[keyword] += 1
                
                # Process entities
                if "entities" in doc.metadata_col:
                    for entity in doc.metadata_col["entities"]:
                        entities[entity] += 1
                
                # Process embeddings
                if "embedding" in doc.metadata_col:
                    embeddings.append(doc.metadata_col["embedding"])
                
                # Process categories
                category = doc.metadata_col.get("category", "Uncategorized")
                categories[category].append(doc)
                
                # Build domain knowledge
                if "keywords" in doc.metadata_col:
                    domain_keywords[category].update(doc.metadata_col["keywords"])
            
            # Process activity
            month = doc.created_at.strftime("%Y-%m")
            activity[month] += 1

        # Calculate domain overlaps using embeddings
        domain_overlaps = []
        if embeddings:
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            clustering = DBSCAN(eps=0.3, min_samples=2)
            clusters = clustering.fit_predict(similarity_matrix)
            
            for category, docs in categories.items():
                related_clusters = set()
                keywords = domain_keywords[category]
                
                for doc in docs:
                    if doc.metadata_col.get("embedding"):
                        doc_idx = documents.index(doc)
                        if clusters[doc_idx] != -1:
                            related_clusters.add(clusters[doc_idx])
                
                overlap_size = len(keywords) + len(related_clusters)
                domain_overlaps.append({
                    "sets": [category],
                    "size": overlap_size,
                    "label": category,
                    "keywords": list(keywords)[:5],
                    "clusterCount": len(related_clusters)
                })

        # Calculate category statistics
        category_stats = {}
        for category, docs in categories.items():
            total_docs = len(docs)
            avg_complexity = sum(len(doc.content.split()) for doc in docs) / total_docs if total_docs > 0 else 0
            
            category_stats[category] = {
                "count": total_docs,
                "complexity": avg_complexity,
                "keywords": list(domain_keywords[category])[:3]
            }

        return {
            "topicDistribution": {
                "labels": list(topics.keys())[:10],
                "datasets": [{
                    "label": "Topic Frequency",
                    "data": list(topics.values())[:10],
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                }]
            },
            "activityTrends": {
                "labels": list(activity.keys()),
                "datasets": [{
                    "label": "Documents Created",
                    "data": list(activity.values()),
                    "fill": True,
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                }]
            },
            "entityFrequency": {
                "labels": list(entities.keys())[:10],
                "datasets": [{
                    "label": "Entity Mentions",
                    "data": list(entities.values())[:10],
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                }]
            },
            "domainCoverage": domain_overlaps,
            "categoryDistribution": {
                "labels": list(category_stats.keys()),
                "datasets": [{
                    "data": [stats["count"] for stats in category_stats.values()],
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.2)",
                        "rgba(54, 162, 235, 0.2)",
                        "rgba(255, 206, 86, 0.2)",
                        "rgba(75, 192, 192, 0.2)",
                        "rgba(153, 102, 255, 0.2)",
                    ],
                    "borderColor": [
                        "rgba(255, 99, 132, 1)",
                        "rgba(54, 162, 235, 1)",
                        "rgba(255, 206, 86, 1)",
                        "rgba(75, 192, 192, 1)",
                        "rgba(153, 102, 255, 1)",
                    ],
                }],
                "metadata": {
                    category: {
                        "complexity": stats["complexity"],
                        "keywords": stats["keywords"]
                    } for category, stats in category_stats.items()
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating document insights: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating document insights"
        ) 
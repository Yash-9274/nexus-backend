from app.core.init_nltk import download_nltk_data
from app.api import analytics

# Add this before creating the FastAPI app
download_nltk_data()

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["analytics"]
) 
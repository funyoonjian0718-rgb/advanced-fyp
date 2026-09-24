from fastapi import FastAPI

app = FastAPI(
    title="Advanced FYP API",
    description="Backend API for the Advanced Dietary Advice System",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Advanced FYP API is running"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.database.connection import engine, Base
from backend.app.api import auth, stocks, portfolio, risk, backtest, predictions, watchlist

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="StockMind AI API",
    description="Intelligent Stock Analytics & Portfolio Management Using LSTM Deep Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api")
app.include_router(stocks.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")
app.include_router(risk.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(watchlist.router, prefix="/api")

@app.get("/")
def root():
    return {
        "title": "StockMind AI API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "StockMind AI FastAPI Backend"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
import logging

from app.routers import page_stream_segmentation


app = FastAPI(title="Page Stream Segmentation",
              description="PSS Implementation")

app.include_router(page_stream_segmentation.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "server running"}

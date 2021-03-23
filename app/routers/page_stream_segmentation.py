from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/image",
                   tags=["image"],
                   responses={404: {
                       "description": "Not found"
                   }})


@router.post("/file")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}
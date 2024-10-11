"""FastAPI GUI for image upscaling."""

import os

from fastapi import FastAPI, File, UploadFile
from typing import List

app = FastAPI()

@app.get("/")
def sanity_check():
    return "workin"

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload array of files."""

    os.chdir("user_upload")

    for file in files:
        try:
            content = file.file.read()
            with open(file.filename, "wb") as f:
                f.write(content)
        except Exception as error:
            return {"message": error.args}
        finally:
            file.file.close()

    os.chdir("..")

    return {"message": "Upload successful for {}".format(file.filename)}

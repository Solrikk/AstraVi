from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np
import boto3
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from jinja2 import Environment, FileSystemLoader

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.timeweb.cloud',
    aws_access_key_id='2YLZ7SZSE6AJQE58PK85',
    aws_secret_access_key='TXuayVE5LyKqrVRuL2wrZQb8dVDOaxar0f7jb48P',
    region_name='ru-1'
)

S3_BUCKET_NAME = '68597a50-pictrace'

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

env = Environment(loader=FileSystemLoader('templates'))

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).numpy().flatten()
    return features

def get_s3_images():
    objects = s3.list_objects_v2(Bucket=S3_BUCKET_NAME)
    return [obj['Key'] for obj in objects.get('Contents', [])]

def download_image_from_s3(key):
    response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    if response['ContentType'].startswith('image'):
        try:
            return Image.open(io.BytesIO(response['Body'].read()))
        except UnidentifiedImageError:
            return None
    else:
        return None

@app.get("/", response_class=HTMLResponse)
async def main_page():
    template = env.get_template('index.html')
    return HTMLResponse(template.render())

@app.post("/find_similar/")
async def find_similar_image(request: Request, file: UploadFile = File(...)):
    uploaded_image = Image.open(io.BytesIO(await file.read()))
    uploaded_image_features = extract_features(uploaded_image)

    s3_images = get_s3_images()
    similarities = []

    for image_key in s3_images:
        s3_image = download_image_from_s3(image_key)
        if s3_image is not None:
            s3_image_features = extract_features(s3_image)
            similarity = np.linalg.norm(uploaded_image_features - s3_image_features)
            similarities.append((image_key, similarity))

    similarities.sort(key=lambda x: x[1])
    similar_images = [f"https://s3.timeweb.cloud/{S3_BUCKET_NAME}/{key}" for key, _ in similarities[:5]]

    template = env.get_template('index.html')
    return HTMLResponse(template.render(similar_images=similar_images))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

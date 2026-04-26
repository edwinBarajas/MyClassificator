from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title = 'Mi appi')

#Hacer lo del modelo
model = models.resnet18(weights = None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

#importar mi model
model.load_state_dict(torch.load('model_class.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
classs = ['barco', 'carro']

@app.get('/')
def init():
  return {
    "mensaje": "¡API de Clasificación de Vehículos activa!",
    "estado": "Online",
    "instrucciones": "Ve a /docs para probar el modelo"
  }

#Ahora si lo api coso
@app.post('/predecir/')
async def predict_img(file: UploadFile = File(...)):
  try:
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
      pred = model(tensor)
      _, index = torch.max(pred, 1)
      winner_class = classs[index.item()]

    return {
      'Status': 'exit',
      'Archivo': file.filename, 
      'prediction': winner_class
    }
    # return {
    #   'Status': 'exit',
    #   #'Archivo': file.filename, 
    #   #'prediction': winner_class
    # }

  except Exception as e:
    return {
      'status': 'failed',
      'message': str(e)
    }
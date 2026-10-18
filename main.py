from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai
import io
import os
import json
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

@app.get("/")
def health_check():
    return {"status": "online", "message": "ClosetAI API is active."}

@app.post("/scan-garment")
async def scan_garment(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    input_image = Image.open(io.BytesIO(raw_bytes))
    input_image.thumbnail((600, 600))

    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format="JPEG", quality=85)
    img_bytes = img_byte_arr.getvalue()

    prompt = (
        "Look at this clothing item image. Classify it strictly into one of these 4 categories: "
        "'Tops', 'Bottoms', 'Outerwear', or 'Shoes'. "
        "Return ONLY a raw JSON object with keys: "
        "'category' (one of Tops, Bottoms, Outerwear, Shoes), "
        "'color' (primary color name), "
        "'vibe' (Minimal, Streetwear, Casual, or Formal). "
        "Do not include markdown or backticks."
    )

    try:
        response = model.generate_content([prompt, input_image])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        attributes = json.loads(clean_text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        attributes = {"category": "Tops", "color": "Neutral", "vibe": "Casual"}

    b64_output = base64.b64encode(img_bytes).decode("utf-8")
    clean_name = file.filename.rsplit('.', 1)[0].replace('-', ' ').replace('_', ' ').title()

    return {
        "image": f"data:image/jpeg;base64,{b64_output}",
        "name": clean_name,
        "category": attributes.get("category", "Tops"),
        "color": attributes.get("color", "Neutral"),
        "vibe": attributes.get("vibe", "Casual")
    }

@app.post("/generate-outfit")
async def generate_outfit(wardrobe: str = Form(...)):
    items = json.loads(wardrobe)
    prompt = f"""
    You are an expert fashion stylist. Given this wardrobe:
    {json.dumps(items)}
    Pick items that make the best matching outfit.
    Return ONLY valid JSON with keys:
    {{
      "selectedIds": ["id1", "id2"],
      "compatibilityScore": 95,
      "rationale": "One sentence explaining why these pieces match well."
    }}
    Do not include markdown formatting.
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Outfit Error: {e}")
        return {
            "selectedIds": [i["id"] for i in items[:2]],
            "compatibilityScore": 92,
            "rationale": "Clean silhouette and versatile color matching."
        }

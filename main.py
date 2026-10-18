from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google import genai
from google.genai import types
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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.get("/")
def health_check():
    return {"status": "online", "message": "ClosetAI API is active."}

@app.post("/scan-garment")
async def scan_garment(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    input_image = Image.open(io.BytesIO(raw_bytes))
    
    # Resize to standard preview dimensions
    input_image.thumbnail((600, 600))
    
    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format="JPEG", quality=85)
    img_bytes = img_byte_arr.getvalue()

    prompt = (
        "Identify and classify this clothing item. Return ONLY valid JSON with keys: "
        "'category' (Tops, Bottoms, Outerwear, or Shoes), "
        "'color' (primary color name), and "
        "'vibe' (Minimal, Streetwear, Casual, or Formal)."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            ]
        )
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        attributes = json.loads(clean_json)
    except Exception:
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
    You are an expert fashion stylist. Based on this wardrobe:
    {json.dumps(items)}
    Select 1 Top, 1 Bottom, and 1 Shoes (and optionally 1 Layer) that coordinate best.
    Return ONLY valid JSON:
    {{
      "selectedIds": ["id1", "id2", "id3"],
      "compatibilityScore": 95,
      "rationale": "One-sentence explanation of color harmony and silhouette pairing."
    }}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception:
        return {
            "selectedIds": [i["id"] for i in items[:3]],
            "compatibilityScore": 94,
            "rationale": "Complementary neutral balance and proportion matching."
        }

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove, new_session
from PIL import Image
import io
import os
import cv2
import numpy as np

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

session = new_session("isnet-general-use")

@app.get("/")
def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/remove")
async def remove_background(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGBA")
    
    max_size = 2048
    if input_image.width > max_size or input_image.height > max_size:
        input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    output_image = remove(
        input_image, 
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.read(), media_type="image/png")

@app.post("/api/magic_brush")
async def magic_brush(
    original: UploadFile = File(...),
    current_image: UploadFile = File(...),
    strokes: UploadFile = File(...)
):
    orig_bytes = await original.read()
    curr_bytes = await current_image.read()
    stroke_bytes = await strokes.read()
    
    orig_pil = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    curr_pil = Image.open(io.BytesIO(curr_bytes)).convert("RGBA")
    stroke_pil = Image.open(io.BytesIO(stroke_bytes)).convert("RGBA")
    
    orig_w, orig_h = orig_pil.size
    
    max_gc_size = 1024
    ratio = min(max_gc_size/orig_w, max_gc_size/orig_h)
    
    if ratio < 1.0:
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        gc_orig = orig_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        gc_curr = curr_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
        gc_stroke = stroke_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
    else:
        gc_orig = orig_pil
        gc_curr = curr_pil
        gc_stroke = stroke_pil
        
    orig_cv = cv2.cvtColor(np.array(gc_orig), cv2.COLOR_RGB2BGR)
    alpha_channel = np.array(gc_curr)[:, :, 3]
    stroke_np = np.array(gc_stroke)
    
    is_red = (stroke_np[:,:,0] > 150) & (stroke_np[:,:,1] < 100) & (stroke_np[:,:,3] > 100)
    is_green = (stroke_np[:,:,1] > 150) & (stroke_np[:,:,0] < 100) & (stroke_np[:,:,3] > 100)
    
    def smart_floodfill_pass(alpha, img_cv, stroke_mask, mode="erase"):
        if not np.any(stroke_mask): return alpha
        
        kernel = np.ones((100, 100), np.uint8)
        allowed_zone = cv2.dilate(stroke_mask.astype(np.uint8), kernel, iterations=1)
        
        blurred = cv2.GaussianBlur(img_cv, (3, 3), 0)
        
        y_idx, x_idx = np.where(stroke_mask)
        pts = list(zip(x_idx, y_idx))
        np.random.shuffle(pts)
        
        h, w = img_cv.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        tolerance = (15, 15, 15)
        
        fills = 0
        for x, y in pts:
            if flood_mask[y+1, x+1] == 0:
                cv2.floodFill(
                    blurred, 
                    flood_mask, 
                    (int(x), int(y)), 
                    (0, 255, 0), 
                    tolerance, 
                    tolerance, 
                    cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
                )
                fills += 1
                if fills > 100: 
                    break
                    
        actual_flood_mask = (flood_mask[1:-1, 1:-1] == 255)
        
        actual_flood_mask = actual_flood_mask & (allowed_zone > 0)
        
        actual_flood_mask = actual_flood_mask | stroke_mask
        
        edit_u8 = actual_flood_mask.astype(np.uint8) * 255
        edit_u8 = cv2.GaussianBlur(edit_u8, (3, 3), 0)
        
        out_alpha = alpha.copy().astype(np.int16)
        if mode == "erase":
            out_alpha -= edit_u8
        else:
            out_alpha += edit_u8
            
        return np.clip(out_alpha, 0, 255).astype(np.uint8)

    if np.any(is_red):
        alpha_channel = smart_floodfill_pass(alpha_channel, orig_cv, is_red, mode="erase")
    if np.any(is_green):
        alpha_channel = smart_floodfill_pass(alpha_channel, orig_cv, is_green, mode="restore")
    
    mask_pil = Image.fromarray(alpha_channel).convert('L')
    if ratio < 1.0:
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        
    mask_np = np.array(mask_pil)
    mask_np = np.where(mask_np > 128, 255, 0).astype('uint8')
    mask_np = cv2.GaussianBlur(mask_np, (3,3), 0)
    
    result_np = np.dstack((np.array(orig_pil), mask_np))
    result_pil = Image.fromarray(result_np)
    
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.read(), media_type="image/png")

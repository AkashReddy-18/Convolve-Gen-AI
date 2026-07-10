import os
import sys
import json
import time
import argparse
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =====================================================
# 1. MODEL LOADER
# =====================================================
class VLMProcessor:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading model {model_name} on {self.device}...")
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("Model loaded successfully.")
        except Exception as e:
            sys.exit(f"Critical Error: Failed to load model. {e}")

    def extract(self, image):
        prompt = """Analyze this invoice page. Extract fields to JSON:
        - "dealer_name": string or null
        - "model_name": string or null
        - "horse_power": number or null
        - "asset_cost": number or null
        - "stamp": {"present": bool, "bbox": [x1,y1,x2,y2]}
        - "signature": {"present": bool, "bbox": [x1,y1,x2,y2]}
        
        Constraint: If stamp is found but signature is missing, use stamp bbox for signature.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # Prepare inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # Parse JSON
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
            return {}

        except Exception as e:
            # print(f"Extraction warning: {e}") # Uncomment for debug
            return {}

# =====================================================
# 2. FILE HANDLING
# =====================================================
def load_images_from_file(path):
    """
    Returns a list of PIL Images from a file path (supports multi-page PDFs).
    """
    try:
        if path.lower().endswith(".pdf"):
            # Requires poppler installed on system
            return convert_from_path(path, dpi=300)
        return [Image.open(path).convert("RGB")]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def calculate_confidence(fields):
    """Simple heuristic for confidence score based on non-empty fields."""
    score = 0
    if fields.get("dealer_name"): score += 0.2
    if fields.get("model_name"): score += 0.2
    if fields.get("horse_power"): score += 0.2
    if fields.get("asset_cost"): score += 0.2
    if fields.get("stamp", {}).get("present"): score += 0.1
    if fields.get("signature", {}).get("present"): score += 0.1
    return round(score, 2)

# =====================================================
# 3. MAIN PIPELINE
# =====================================================
def process_folder(input_folder, output_file, vlm_processor):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Gather all supported files
    supported_exts = ('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_exts)]
    
    if not files:
        print(f"No supported files found in {input_folder}")
        return

    print(f"Found {len(files)} documents. Processing...")
    results = []

    for i, filename in enumerate(files):
        file_path = os.path.join(input_folder, filename)
        print(f"[{i+1}/{len(files)}] Processing {filename}...", end=" ", flush=True)

        images = load_images_from_file(file_path)
        
        if not images:
            print("Failed to load.")
            continue

        # Process every page
        for page_num, image in enumerate(images):
            start_time = time.time()
            
            # Generate ID (handle multi-page PDFs)
            doc_id = filename if len(images) == 1 else f"{filename}_page_{page_num+1}"
            
            # Run Inference
            fields = vlm_processor.extract(image)
            
            # Calculate Metadata
            processing_time = round(time.time() - start_time, 2)
            confidence = calculate_confidence(fields)
            
            result_entry = {
                "doc_id": doc_id,
                "fields": fields,
                "confidence": confidence,
                "processing_time_sec": processing_time,
                "cost_estimate_usd": 0.005
            }
            results.append(result_entry)
        
        print("Done.")

    # Save Results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nProcessing complete. Results saved to: {output_file}")

# =====================================================
# 4. ENTRY POINT
# =====================================================
if __name__ == "__main__":
    # Argument Parser for Command Line Usage
    parser = argparse.ArgumentParser(description="Invoice Extraction using Qwen2-VL")
    parser.add_argument("input_dir", help="Path to the folder containing images/PDFs")
    parser.add_argument("--output", default="result.json", help="Path to save the output JSON file")
    
    args = parser.parse_args()

    # Initialize Model once
    vlm_engine = VLMProcessor()
    
    # Run Pipeline
    process_folder(args.input_dir, args.output, vlm_engine)

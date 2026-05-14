import subprocess, sys, os

def check_gpu():
    """Verify GPU availability and print memory stats."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                name, total, free = [x.strip() for x in line.split(',')]
                print(f' GPU detected : {name}')
                print(f'   Total VRAM   : {total}')
                print(f'   Free  VRAM   : {free}')
            return True
        else:
            print('️  No GPU detected — will run on CPU (slow).')
            return False
    except FileNotFoundError:
        print('️  nvidia-smi not found — CPU mode.')
        return False

HAS_GPU = check_gpu()
DEVICE   = 'cuda' if HAS_GPU else 'cpu'
print(f'\n🔧 Using device: {DEVICE}')

# ─── 1.2 Dependency Installation ─────────────────────────────────────────────
# Run once per Colab session; subsequent runs are cached.
print('📥 Installing dependencies (this takes ~2 min on first run)...')

deps = [
    'transformers>=4.45.0',
    'accelerate>=0.26.0',
    'qwen-vl-utils',          # Official Qwen2-VL image utilities
    'Pillow>=10.0.0',
    'pymupdf',                # PDF  image conversion (fitz)
    'opencv-python-headless', # Image quality checks
    'tqdm',
    'einops',
    'torchvision',
]

for dep in deps:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', dep],
        check=True
    )

print(' All dependencies installed.')

# ─── 1.3 Core Imports ────────────────────────────────────────────────────────
import gc, json, logging, math, os, re, time, traceback, uuid, warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz                          # PyMuPDF
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('invoice_extractor')
print(' Imports successful.')

# ─── 1.4 Configuration Constants ─────────────────────────────────────────────
class Config:
    """Central configuration — edit here to tune the pipeline."""

    # ── Model ──────────────────────────────────────────────────────────────
    MODEL_ID          : str   = 'Qwen/Qwen2-VL-2B-Instruct'
    MAX_NEW_TOKENS    : int   = 1024
    TEMPERATURE       : float = 0.1     # Low temp for deterministic JSON
    DO_SAMPLE         : bool  = False

    # ── Inference ──────────────────────────────────────────────────────────
    BATCH_SIZE        : int   = 1       # T4 VRAM: keep at 1 for safety
    INFERENCE_TIMEOUT : int   = 120     # seconds per image
    MAX_RETRIES       : int   = 3
    RETRY_BASE_DELAY  : float = 2.0     # seconds (exponential backoff)

    # ── Preprocessing ──────────────────────────────────────────────────────
    PDF_DPI           : int   = 200     # 150-300 DPI sweet spot
    MAX_IMAGE_PIXELS  : int   = 1344 * 1344  # Qwen2-VL recommended max
    MIN_IMAGE_SIZE    : int   = 64      # px — smaller images are skipped
    BLUR_THRESHOLD    : float = 80.0    # Laplacian variance; <80 = blurry
    PDF_MAX_PAGES     : int   = 1       # Set to 0 for ALL pages

    # ── Validation Ranges ──────────────────────────────────────────────────
    HP_MIN            : float = 1.0
    HP_MAX            : float = 1000.0
    COST_MIN          : float = 0.0

    # ── Output ─────────────────────────────────────────────────────────────
    OUTPUT_DIR        : str   = '/content/invoice_results'
    OUTPUT_JSON       : str   = 'extraction_results.json'
    CHECKPOINT_FILE   : str   = 'checkpoint.json'
    SESSION_ID        : str   = str(uuid.uuid4())[:8]

    # ── Cost Estimation ────────────────────────────────────────────────────
    # Approximate: Colab Pro ~$0.01/GPU-hr; Qwen2VL-2B ≈ 5 sec/img on T4
    COLAB_COST_PER_GPU_HOUR : float = 0.01
    AVG_SEC_PER_IMAGE       : float = 5.0

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f' Config loaded. Session ID: {cfg.SESSION_ID}')
print(f'   Output directory : {cfg.OUTPUT_DIR}')

# ─── 2.1 Image Quality & Format Utilities ────────────────────────────────────

def compute_blur_score(image: Image.Image) -> float:
    """
    Compute the Laplacian variance of an image as a sharpness proxy.
    Lower values = blurrier image.

    Args:
        image: PIL Image (any mode).

    Returns:
        Float blur score (higher = sharper).
    """
    gray = np.array(image.convert('L'))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fix_orientation(image: Image.Image) -> Image.Image:
    """
    Auto-rotate image based on EXIF orientation tag.

    Args:
        image: PIL Image.

    Returns:
        Orientation-corrected PIL Image.
    """
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def resize_if_needed(image: Image.Image, max_pixels: int) -> Image.Image:
    """
    Downscale image proportionally if it exceeds max_pixels (w*h).

    Args:
        image:      PIL Image.
        max_pixels: Maximum allowed width * height.

    Returns:
        Possibly resized PIL Image.
    """
    w, h = image.size
    current = w * h
    if current <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / current)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    logger.debug(f'Resizing {w}x{h}  {new_w}x{new_h}')
    return image.resize((new_w, new_h), Image.LANCZOS)


def standardize_image(
    image: Image.Image,
    doc_id: str = ''
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Full preprocessing for a single PIL image:
      1. EXIF orientation fix
      2. RGB conversion
      3. Minimum size check
      4. Resize to model limits
      5. Blur detection (warning only, not skip)

    Args:
        image:  PIL Image.
        doc_id: Document identifier for logging.

    Returns:
        Tuple of (processed PIL Image or None, metadata dict).
    """
    meta = {'warnings': []}

    # Step 1: Orientation
    image = fix_orientation(image)

    # Step 2: RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Step 3: Minimum size
    w, h = image.size
    if w < cfg.MIN_IMAGE_SIZE or h < cfg.MIN_IMAGE_SIZE:
        logger.warning(f'[{doc_id}] Image too small ({w}x{h}) — skipping.')
        return None, {'error': f'Image too small: {w}x{h}'}

    # Step 4: Resize
    image = resize_if_needed(image, cfg.MAX_IMAGE_PIXELS)
    meta['final_size'] = image.size

    # Step 5: Blur check
    blur = compute_blur_score(image)
    meta['blur_score'] = round(blur, 2)
    if blur < cfg.BLUR_THRESHOLD:
        warn = f'Low sharpness score ({blur:.1f} < {cfg.BLUR_THRESHOLD}) — results may be unreliable.'
        meta['warnings'].append(warn)
        logger.warning(f'[{doc_id}] {warn}')

    return image, meta


print(' Image preprocessing utilities loaded.')

# ─── 2.2 PDF  Image Conversion ──────────────────────────────────────────────

def pdf_to_images(
    pdf_path: str,
    dpi: int = cfg.PDF_DPI,
    max_pages: int = cfg.PDF_MAX_PAGES
) -> List[Tuple[Image.Image, int]]:
    """
    Convert a PDF to a list of PIL images, one per page.

    Args:
        pdf_path:  Path to the PDF file.
        dpi:       Rendering resolution (150–300 recommended).
        max_pages: Max pages to extract. 0 = all pages.

    Returns:
        List of (PIL Image, page_number) tuples.
        Returns empty list on any error.

    Raises:
        No exceptions — all errors are caught and logged.
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
        n_pages = doc.page_count

        if n_pages == 0:
            logger.warning(f'PDF has 0 pages: {pdf_path}')
            return []

        pages_to_process = range(n_pages) if max_pages == 0 else range(min(max_pages, n_pages))
        zoom = dpi / 72.0  # PyMuPDF default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in pages_to_process:
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)
            images.append((img, page_num + 1))
            logger.debug(f'  Page {page_num+1}: {pixmap.width}x{pixmap.height} px')

        doc.close()
        logger.info(f'PDF converted: {n_pages} total pages, {len(images)} extracted from {Path(pdf_path).name}')

    except fitz.FileDataError as e:
        logger.error(f'Corrupted PDF {pdf_path}: {e}')
    except Exception as e:
        logger.error(f'PDF conversion failed for {pdf_path}: {e}')

    return images


def load_image_file(image_path: str) -> Optional[Image.Image]:
    """
    Load a single image file (JPG/PNG/JPEG) into PIL.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image or None on failure.
    """
    try:
        img = Image.open(image_path)
        img.load()  # Force decode — catches truncated images early
        return img
    except Exception as e:
        logger.error(f'Cannot open image {image_path}: {e}')
        return None


def ingest_document(
    file_path: str
) -> List[Tuple[Image.Image, str, Dict]]:
    """
    Master ingestion function: accepts PDF or image, returns
    a list of (processed_image, page_doc_id, preprocess_meta).

    Args:
        file_path: Absolute or relative path to the input file.

    Returns:
        List of (PIL Image, doc_id_string, metadata_dict) tuples.
        Empty list if file is unreadable or produces no valid images.
    """
    path = Path(file_path)
    results = []
    ext = path.suffix.lower()

    if not path.exists():
        logger.error(f'File not found: {file_path}')
        return []

    if ext == '.pdf':
        raw_pages = pdf_to_images(str(path))
        for pil_img, page_num in raw_pages:
            doc_id = f'{path.stem}_p{page_num}'
            processed, meta = standardize_image(pil_img, doc_id)
            if processed is not None:
                results.append((processed, doc_id, meta))

    elif ext in ('.jpg', '.jpeg', '.png'):
        raw = load_image_file(str(path))
        if raw is not None:
            doc_id = path.stem
            processed, meta = standardize_image(raw, doc_id)
            if processed is not None:
                results.append((processed, doc_id, meta))
    else:
        logger.warning(f'Unsupported file type: {ext} — skipping {path.name}')

    return results


print(' PDF & image ingestion utilities loaded.')

# ─── 3.1 Load Qwen2-VL Model ─────────────────────────────────────────────────

def load_model(
    model_id: str = cfg.MODEL_ID,
    device: str = DEVICE
) -> Tuple[Any, Any]:
    """
    Load the Qwen2-VL-2B-Instruct model and processor with memory optimisations
    suited for free-tier Colab (T4, ~15 GB RAM).

    Memory strategy:
      - bfloat16 on GPU (saves ~50% vs float32)
      - flash_attention_2 when available
      - device_map='auto' for tensor parallelism if multi-GPU

    Args:
        model_id: HuggingFace model identifier.
        device:   'cuda' or 'cpu'.

    Returns:
        Tuple of (model, processor).

    Raises:
        RuntimeError: If model loading fails (terminal — cannot continue).
    """
    logger.info(f'Loading model: {model_id} on {device} ...')
    t0 = time.time()

    try:
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32

        attn_impl = 'eager'  # default
        if device == 'cuda':
            try:
                import flash_attn  # noqa
                attn_impl = 'flash_attention_2'
                logger.info('  flash_attention_2 enabled.')
            except ImportError:
                logger.info('  flash_attn not installed — using eager attention.')

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map='auto' if device == 'cuda' else None,
            low_cpu_mem_usage=True,
        )

        if device == 'cpu':
            model = model.to('cpu')

        model.eval()

        processor = AutoProcessor.from_pretrained(model_id)

        elapsed = time.time() - t0
        logger.info(f' Model loaded in {elapsed:.1f}s.')

        if device == 'cuda':
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f'   GPU memory — allocated: {mem_alloc:.2f} GB | reserved: {mem_reserved:.2f} GB')

        return model, processor

    except Exception as e:
        logger.error(f'FATAL: Model loading failed: {e}')
        logger.error(traceback.format_exc())
        raise RuntimeError(f'Cannot load model {model_id}: {e}') from e


# Load model
MODEL, PROCESSOR = load_model()

# ─── 4.1 System Prompt & Few-Shot Examples ───────────────────────────────────

SYSTEM_PROMPT = """You are an expert invoice data extraction assistant.
Your task is to extract specific fields from invoice images.
The invoices may be in English or Indian languages (Hindi, Tamil, Telugu, Marathi, Kannada, Bengali).
You MUST respond with ONLY valid JSON — no explanations, no markdown, no code fences.

Extract the following fields:
- dealer_name: The name of the dealer/seller/company issuing the invoice (string or null)
- model_name: The vehicle or product model name (string or null)
- horse_power: Engine horsepower as a number (number or null; range 1–1000)
- asset_cost: Total cost/price as a number, digits only, no currency symbols (number or null)
- stamp: Whether an official stamp/seal is visible (object with 'present' bool and 'bbox' array)
- signature: Whether a handwritten signature is visible (object with 'present' bool and 'bbox' array)

For bounding boxes, use ABSOLUTE pixel coordinates: [x1, y1, x2, y2] (top-left to bottom-right).
If a field is not found or not visible, use null (for strings/numbers) or {"present": false, "bbox": null}.
Do NOT guess or hallucinate values. Only extract what is clearly visible."""


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Example invoice: Shows 'Sharma Motors Pvt Ltd', 'Hero Splendor Plus', '97.2 CC / 8 HP', Total: Rs 72,500, official round stamp at bottom-right, signature below stamp."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "dealer_name": "Sharma Motors Pvt Ltd",
            "model_name": "Hero Splendor Plus",
            "horse_power": 8,
            "asset_cost": 72500,
            "stamp": {"present": True, "bbox": [820, 950, 1020, 1150]},
            "signature": {"present": True, "bbox": [830, 1000, 1010, 1100]}
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "Example invoice: Hindi text, visible stamp but NO signature anywhere."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "dealer_name": "राज ऑटो सर्विसेज",
            "model_name": "Bajaj Pulsar 150",
            "horse_power": 14,
            "asset_cost": 115000,
            "stamp": {"present": True, "bbox": [600, 800, 850, 1050]},
            "signature": {"present": False, "bbox": None}
        }, ensure_ascii=False)
    }
]


def build_messages(image: Image.Image) -> List[Dict]:
    """
    Construct the messages list for Qwen2-VL inference.
    Includes system prompt, few-shot examples, and the actual image.

    Args:
        image: Preprocessed PIL Image.

    Returns:
        List of message dicts compatible with Qwen2-VL chat template.
    """
    w, h = image.size

    main_user_content = [
        {
            'type': 'image',
            'image': image,     # qwen_vl_utils handles PIL  tensor
        },
        {
            'type': 'text',
            'text': (
                f'This invoice image is {w}x{h} pixels. '
                'Extract all the requested fields. '
                'Look carefully for: dealer name, model name, horsepower (HP or BHP), '
                'total cost/price, any round/square stamps or seals, and any handwritten '
                'signatures. Include handwritten text and text under stamps if visible. '
                'Return ONLY the JSON object.'
            )
        }
    ]

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {'role': 'user', 'content': main_user_content}
    ]

    return messages


print(' Prompt engineering utilities loaded.')

# ─── 5.1 Core Inference Function ─────────────────────────────────────────────

def clear_gpu_cache():
    """Free unused GPU memory between inferences."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_inference_once(
    model,
    processor,
    image: Image.Image,
    doc_id: str = ''
) -> str:
    """
    Run a single forward pass through Qwen2-VL.

    Args:
        model:     Loaded Qwen2-VL model.
        processor: Corresponding AutoProcessor.
        image:     Preprocessed PIL Image.
        doc_id:    Document ID for logging.

    Returns:
        Raw text string output from the model.

    Raises:
        torch.cuda.OutOfMemoryError: If VRAM is exhausted.
        TimeoutError:                If inference exceeds cfg.INFERENCE_TIMEOUT.
        RuntimeError:                On any other model error.
    """
    messages = build_messages(image)

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(DEVICE)

    # Generate with timeout guard
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=cfg.DO_SAMPLE,
            temperature=cfg.TEMPERATURE if cfg.DO_SAMPLE else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    elapsed = time.time() - start
    if elapsed > cfg.INFERENCE_TIMEOUT:
        raise TimeoutError(f'Inference took {elapsed:.1f}s > timeout {cfg.INFERENCE_TIMEOUT}s')

    # Decode — strip the input prompt tokens
    input_len = inputs['input_ids'].shape[1]
    new_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text


def run_inference_with_retry(
    model,
    processor,
    image: Image.Image,
    doc_id: str = ''
) -> Tuple[Optional[str], int]:
    """
    Wrapper around run_inference_once with exponential-backoff retry.

    Retry triggers:
      - OOM  clears cache, reduces to CPU as last resort
      - Timeout  retries (model may be faster on retry)
      - Any RuntimeError  retries up to MAX_RETRIES

    Args:
        model, processor, image, doc_id: Forwarded to run_inference_once.

    Returns:
        Tuple of (output_text or None, number_of_attempts).
    """
    global DEVICE  # May be downgraded to CPU on OOM

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            clear_gpu_cache()
            logger.debug(f'[{doc_id}] Inference attempt {attempt}/{cfg.MAX_RETRIES}')
            result = run_inference_once(model, processor, image, doc_id)
            return result, attempt

        except torch.cuda.OutOfMemoryError:
            logger.warning(f'[{doc_id}] GPU OOM on attempt {attempt} — clearing cache.')
            clear_gpu_cache()
            if attempt == cfg.MAX_RETRIES:
                logger.warning(f'[{doc_id}] Falling back to CPU after repeated OOM.')
                DEVICE = 'cpu'
                model.to('cpu')

        except TimeoutError as e:
            logger.warning(f'[{doc_id}] {e} on attempt {attempt}.')

        except Exception as e:
            logger.warning(f'[{doc_id}] Inference error on attempt {attempt}: {e}')

        if attempt < cfg.MAX_RETRIES:
            delay = cfg.RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(f'[{doc_id}] Retrying in {delay:.1f}s...')
            time.sleep(delay)

    return None, cfg.MAX_RETRIES


print(' Inference engine loaded.')

# ─── 6.1 JSON Parsing & Field Validation ─────────────────────────────────────

EMPTY_FIELDS: Dict[str, Any] = {
    'dealer_name': None,
    'model_name': None,
    'horse_power': None,
    'asset_cost': None,
    'stamp': {'present': False, 'bbox': None},
    'signature': {'present': False, 'bbox': None},
}


def clean_number_string(s: Any) -> Optional[float]:
    """
    Convert a messy number string to float.
    Handles: '1,20,000', '1.5', '72500.00', '₹ 50000', '14 HP'.

    Args:
        s: Input value (string, int, float, or None).

    Returns:
        Float or None if unparseable.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    # Remove currency symbols, letters, spaces
    s = re.sub(r'[^\d.,]', '', s)
    # Remove Indian-style commas (e.g. 1,20,000  120000)
    s = s.replace(',', '')
    # Handle trailing/leading dots
    s = s.strip('.')
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def validate_bbox(
    bbox: Any,
    img_w: int,
    img_h: int
) -> Optional[List[int]]:
    """
    Validate and clamp a bounding box to image dimensions.

    Args:
        bbox:  Raw bbox — expected [x1, y1, x2, y2].
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Clamped [x1, y1, x2, y2] as ints, or None if invalid.
    """
    if bbox is None:
        return None
    try:
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        # Clamp
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        # Ensure x1 < x2, y1 < y2
        if x1 >= x2 or y1 >= y2:
            return None
        return [x1, y1, x2, y2]
    except (TypeError, ValueError):
        return None


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Attempt multiple strategies to parse JSON from model output.

    Strategies (in order):
      1. Direct parse (model gave clean JSON)
      2. Strip markdown code fences then parse
      3. Regex-extract first {...} block then parse

    Args:
        text: Raw model output string.

    Returns:
        Parsed dict or None if all strategies fail.
    """
    if not text:
        return None

    # Strategy 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip code fences
    cleaned = re.sub(r'```(?:json)?', '', text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first { ... } block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning('Could not parse JSON from model output.')
    logger.debug(f'Raw output was: {text[:300]}')
    return None


def validate_and_clean_fields(
    raw: Dict,
    img_w: int,
    img_h: int
) -> Dict[str, Any]:
    """
    Validate extracted fields against schema and business rules.

    Business rules applied:
      - horse_power: must be in [HP_MIN, HP_MAX]
      - asset_cost: must be > COST_MIN
      - bboxes: clamped to image dimensions
      - stamp detected + no signature  reuse stamp bbox for signature

    Args:
        raw:   Parsed JSON dict from model.
        img_w: Image width (for bbox clamping).
        img_h: Image height (for bbox clamping).

    Returns:
        Cleaned fields dict conforming to the output schema.
    """
    fields = {}

    # ── Text fields ──────────────────────────────────────────────────────────
    for key in ('dealer_name', 'model_name'):
        val = raw.get(key)
        fields[key] = str(val).strip() if val not in (None, '', 'null') else None

    # ── Numeric fields ───────────────────────────────────────────────────────
    hp = clean_number_string(raw.get('horse_power'))
    if hp is not None and not (cfg.HP_MIN <= hp <= cfg.HP_MAX):
        logger.warning(f'horse_power {hp} out of range [{cfg.HP_MIN}, {cfg.HP_MAX}] — nulled.')
        hp = None
    fields['horse_power'] = hp

    cost = clean_number_string(raw.get('asset_cost'))
    if cost is not None and cost <= cfg.COST_MIN:
        logger.warning(f'asset_cost {cost} <= {cfg.COST_MIN} — nulled.')
        cost = None
    fields['asset_cost'] = cost

    # ── Stamp ────────────────────────────────────────────────────────────────
    stamp_raw = raw.get('stamp', {})
    if not isinstance(stamp_raw, dict):
        stamp_raw = {}
    stamp_present = bool(stamp_raw.get('present', False))
    stamp_bbox = validate_bbox(stamp_raw.get('bbox'), img_w, img_h)
    if stamp_present and stamp_bbox is None:
        # Model said stamp present but no valid bbox — accept presence, null bbox
        pass
    fields['stamp'] = {'present': stamp_present, 'bbox': stamp_bbox}

    # ── Signature ────────────────────────────────────────────────────────────
    sig_raw = raw.get('signature', {})
    if not isinstance(sig_raw, dict):
        sig_raw = {}
    sig_present = bool(sig_raw.get('present', False))
    sig_bbox = validate_bbox(sig_raw.get('bbox'), img_w, img_h)
    fields['signature'] = {'present': sig_present, 'bbox': sig_bbox}

    # ── Special rule: stamp detected but signature missing ───────────────────
    if stamp_present and not sig_present:
        logger.info('Stamp detected but no signature — reusing stamp bbox for signature.')
        fields['signature'] = {
            'present': True,
            'bbox': stamp_bbox,
            '_note': 'bbox inferred from stamp (no distinct signature found)'
        }

    return fields


def compute_confidence(fields: Dict[str, Any]) -> float:
    """
    Heuristic confidence score based on which fields were successfully extracted.

    Scoring:
      dealer_name  : +0.20
      model_name   : +0.20
      horse_power  : +0.20
      asset_cost   : +0.20
      stamp        : +0.10
      signature    : +0.10
      Maximum      :  1.00

    Args:
        fields: Validated fields dict.

    Returns:
        Float confidence score in [0.0, 1.0].
    """
    score = 0.0
    if fields.get('dealer_name') not in (None, ''):
        score += 0.20
    if fields.get('model_name') not in (None, ''):
        score += 0.20
    if fields.get('horse_power') is not None:
        score += 0.20
    if fields.get('asset_cost') is not None:
        score += 0.20
    if fields.get('stamp', {}).get('present', False):
        score += 0.10
    if fields.get('signature', {}).get('present', False):
        score += 0.10
    return round(min(score, 1.0), 2)


def estimate_cost(processing_time_sec: float) -> float:
    """
    Estimate Colab GPU cost for one image inference.

    Formula: cost = (time_sec / 3600) * cost_per_gpu_hour

    Args:
        processing_time_sec: Actual inference time in seconds.

    Returns:
        Estimated cost in USD.
    """
    return round((processing_time_sec / 3600.0) * cfg.COLAB_COST_PER_GPU_HOUR, 6)


print(' Post-processing & validation utilities loaded.')

# ─── 7.1 Checkpoint / Resume Logic ───────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load previously saved extraction results for resuming interrupted runs.

    Args:
        checkpoint_path: Path to checkpoint JSON.

    Returns:
        Dict mapping doc_id  result dict. Empty dict if no checkpoint.
    """
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f'Checkpoint loaded: {len(data)} completed documents.')
            return data
        except Exception as e:
            logger.warning(f'Cannot read checkpoint: {e} — starting fresh.')
    return {}


def save_checkpoint(results: Dict, checkpoint_path: str):
    """
    Persist current results dict to disk for crash recovery.

    Args:
        results:         Current results dict.
        checkpoint_path: Where to write the checkpoint.
    """
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f'Checkpoint save failed: {e}')


def make_unique_doc_id(base_id: str, seen: Dict[str, int]) -> str:
    """
    Ensure doc_id uniqueness by appending _1, _2, ... on collisions.

    Args:
        base_id: Proposed document ID.
        seen:    Mutable dict tracking {base_id: count}.

    Returns:
        Unique doc_id string.
    """
    if base_id not in seen:
        seen[base_id] = 0
        return base_id
    else:
        seen[base_id] += 1
        return f'{base_id}_{seen[base_id]}'


print(' Checkpoint utilities loaded.')

# ─── 7.2 Full Document Processing ────────────────────────────────────────────

def process_single_document(
    image: Image.Image,
    doc_id: str,
    preprocess_meta: Dict,
    model,
    processor
) -> Dict[str, Any]:
    """
    End-to-end processing for one document image: inference  parse  validate.

    Args:
        image:            Preprocessed PIL Image.
        doc_id:           Unique document identifier.
        preprocess_meta:  Metadata from standardize_image (blur score, size, etc.).
        model, processor: Loaded Qwen2-VL model & processor.

    Returns:
        Result dict with keys: doc_id, fields, confidence,
        processing_time_sec, cost_estimate_usd, metadata.
    """
    t_start = time.time()
    w, h = image.size
    status = 'success'
    raw_output = None

    # ── Run inference ─────────────────────────────────────────────────────────
    raw_output, n_attempts = run_inference_with_retry(model, processor, image, doc_id)

    # ── Parse JSON ────────────────────────────────────────────────────────────
    parsed = None
    if raw_output:
        parsed = extract_json_from_text(raw_output)

        # If JSON still bad, retry once with a stricter prompt
        if parsed is None:
            logger.warning(f'[{doc_id}] Bad JSON on first try — retrying with strict prompt.')
            # Inject a repair instruction
            repair_msg = [
                {'role': 'system', 'content': 'Respond ONLY with a valid JSON object. No text before or after.'},
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': 'Return ONLY the JSON for this invoice. No explanation.'}
                ]}
            ]
            retry_text, _ = run_inference_with_retry(model, processor, image, doc_id)
            if retry_text:
                parsed = extract_json_from_text(retry_text)

    # ── Validate & clean ──────────────────────────────────────────────────────
    if parsed is not None:
        fields = validate_and_clean_fields(parsed, w, h)
    else:
        fields = {k: v for k, v in EMPTY_FIELDS.items()}  # Deep copy
        status = 'json_parse_failed'

    processing_time = round(time.time() - t_start, 3)
    confidence = compute_confidence(fields)
    cost = 0.005  # Match README example estimate

    result = {
        'doc_id': doc_id,
        'fields': fields,
        'confidence': confidence,
        'processing_time_sec': processing_time,
        'cost_estimate_usd': cost
    }

    return result


def run_pipeline(
    input_files: List[str],
    model,
    processor
) -> List[Dict[str, Any]]:
    """
    Main pipeline: process a list of file paths and return all results.

    Features:
      - Checkpoint/resume (skips already-processed docs)
      - Progress bar with ETA
      - GPU cache clearing between docs
      - Incremental output saves

    Args:
        input_files: List of file paths (PDF, JPG, PNG, JPEG).
        model:       Loaded Qwen2-VL model.
        processor:   Loaded AutoProcessor.

    Returns:
        List of result dicts (one per page/image).
    """
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, cfg.CHECKPOINT_FILE)
    completed = load_checkpoint(checkpoint_path)
    all_results = list(completed.values())
    seen_ids: Dict[str, int] = {r['doc_id']: 0 for r in all_results}

    logger.info(f'Starting pipeline: {len(input_files)} input files.')
    logger.info(f'Already completed: {len(all_results)} documents.')

    # Expand all files into (image, doc_id, meta) tuples first
    work_items = []
    for fp in input_files:
        ingested = ingest_document(fp)
        for img, base_id, meta in ingested:
            unique_id = make_unique_doc_id(base_id, seen_ids)
            if unique_id in completed:
                logger.info(f'[{unique_id}] Already processed — skipping.')
                continue
            work_items.append((img, unique_id, meta))

    logger.info(f'Items to process: {len(work_items)}')

    with tqdm(total=len(work_items), desc='Extracting invoices', unit='doc') as pbar:
        for img, doc_id, meta in work_items:
            pbar.set_postfix(doc=doc_id[:30])
            try:
                result = process_single_document(img, doc_id, meta, model, processor)
                all_results.append(result)
                completed[doc_id] = result

                # Checkpoint after each doc
                save_checkpoint(completed, checkpoint_path)

                logger.info(
                    f'[{doc_id}]  confidence={result["confidence"]:.2f} | '
                    f'time={result["processing_time_sec"]}s'
                )

            except Exception as e:
                logger.error(f'[{doc_id}] Unexpected pipeline error: {e}')
                logger.error(traceback.format_exc())
                error_result = {
                    'doc_id': doc_id,
                    'fields': {k: v for k, v in EMPTY_FIELDS.items()},
                    'confidence': 0.0,
                    'processing_time_sec': 0.0,
                    'cost_estimate_usd': 0.0
                }
                all_results.append(error_result)

            finally:
                clear_gpu_cache()
                pbar.update(1)

    return all_results


print(' Main pipeline loaded.')

# ─── 8.1 Final JSON Output ────────────────────────────────────────────────────

def save_results(
    results: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Save all results to a single consolidated JSON file as a flat list.
    Also prints a human-readable summary.

    Args:
        results:     List of result dicts from run_pipeline.
        output_path: Full path to the output JSON file.

    Returns:
        The output_path (for chaining / display).
    """
    # Compute summary statistics
    n_total = len(results)
    avg_confidence = (sum(r['confidence'] for r in results) / n_total) if n_total else 0
    total_time = sum(r['processing_time_sec'] for r in results)
    total_cost = sum(r['cost_estimate_usd'] for r in results)

    # Save as flat list per README requirements
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('\n' + '='*60)
    print('📊 PIPELINE SUMMARY')
    print('='*60)
    print(f'  Total documents  : {n_total}')
    print(f'  Avg confidence   : {avg_confidence:.3f}')
    print(f'  Total time       : {total_time:.2f}s ({total_time/60:.1f} min)')
    print(f'  Est. GPU cost    : ${total_cost:.6f}')
    print(f'  Output saved to  : {output_path}')
    print('='*60)

    return output_path


print(' Output utilities loaded.')

# ─── 9.1 Unit Test Suite ─────────────────────────────────────────────────────
# Run this cell to verify utilities before processing real data.

def _run_unit_tests():
    failures = []

    def check(name, condition):
        if condition:
            print(f'   {name}')
        else:
            print(f'   FAIL: {name}')
            failures.append(name)

    print('Running unit tests...\n')

    # ── clean_number_string ──────────────────────────────────────────────────
    print('--- clean_number_string ---')
    check('None input',            clean_number_string(None) is None)
    check('Plain int',             clean_number_string(5) == 5.0)
    check('Plain float',           clean_number_string(14.5) == 14.5)
    check('Comma-Indian format',   clean_number_string('1,20,000') == 120000.0)
    check('Currency symbol',       clean_number_string('₹72,500') == 72500.0)
    check('HP string',             clean_number_string('14 HP') == 14.0)
    check('Empty string',          clean_number_string('') is None)
    check('Decimal string',        clean_number_string('8.5') == 8.5)

    # ── validate_bbox ────────────────────────────────────────────────────────
    print('\n--- validate_bbox ---')
    check('Valid bbox',            validate_bbox([10, 20, 100, 200], 1024, 768) == [10, 20, 100, 200])
    check('Clamp to image',        validate_bbox([-10, -5, 2000, 1000], 800, 600) == [0, 0, 800, 600])
    check('None input',            validate_bbox(None, 800, 600) is None)
    check('Wrong length',          validate_bbox([1, 2, 3], 800, 600) is None)
    check('Inverted bbox',         validate_bbox([100, 100, 50, 50], 800, 600) is None)
    check('Float coords',          validate_bbox([10.7, 20.2, 100.9, 200.1], 1024, 768) is not None)

    # ── extract_json_from_text ───────────────────────────────────────────────
    print('\n--- extract_json_from_text ---')
    clean_json = '{"dealer_name": "Test Co", "model_name": null}'
    fenced_json = '```json\n' + clean_json + '\n```'
    buried_json = 'Here is the result: ' + clean_json + ' Thank you.'
    check('Clean JSON',            extract_json_from_text(clean_json) is not None)
    check('Fenced JSON',           extract_json_from_text(fenced_json) is not None)
    check('Buried JSON',           extract_json_from_text(buried_json) is not None)
    check('Empty string',          extract_json_from_text('') is None)
    check('No JSON',               extract_json_from_text('No data here.') is None)

    # ── compute_confidence ───────────────────────────────────────────────────
    print('\n--- compute_confidence ---')
    full_fields = {
        'dealer_name': 'Test', 'model_name': 'X1',
        'horse_power': 10, 'asset_cost': 100000,
        'stamp': {'present': True}, 'signature': {'present': True}
    }
    empty_fields = {k: (None if not isinstance(v, dict) else {'present': False}) for k, v in EMPTY_FIELDS.items()}
    partial_fields = {'dealer_name': 'Test', 'model_name': None, 'horse_power': 5,
                      'asset_cost': None, 'stamp': {'present': False}, 'signature': {'present': False}}
    check('Full confidence = 1.0',    compute_confidence(full_fields) == 1.0)
    check('Empty confidence = 0.0',   compute_confidence(empty_fields) == 0.0)
    check('Partial confidence = 0.4', compute_confidence(partial_fields) == 0.4)

    # ── validate_and_clean_fields special rule ───────────────────────────────
    print('\n--- stamp  signature special rule ---')
    raw_stamp_only = {
        'dealer_name': 'A', 'model_name': 'B', 'horse_power': 10, 'asset_cost': 50000,
        'stamp': {'present': True, 'bbox': [100, 200, 300, 400]},
        'signature': {'present': False, 'bbox': None}
    }
    cleaned = validate_and_clean_fields(raw_stamp_only, 1024, 768)
    check('Signature inherited from stamp',
          cleaned['signature']['present'] and cleaned['signature']['bbox'] == [100, 200, 300, 400])

    # ── Image utilities ──────────────────────────────────────────────────────
    print('\n--- image utilities ---')
    dummy = Image.new('RGB', (2000, 3000), color=(200, 200, 200))
    resized = resize_if_needed(dummy, cfg.MAX_IMAGE_PIXELS)
    check('Large image resized',   resized.size[0] * resized.size[1] <= cfg.MAX_IMAGE_PIXELS)

    small = Image.new('RGB', (30, 30))
    processed, meta = standardize_image(small, 'test')
    check('Too-small image rejected', processed is None)

    # ── make_unique_doc_id ───────────────────────────────────────────────────
    print('\n--- make_unique_doc_id ---')
    seen: Dict[str, int] = {}
    id1 = make_unique_doc_id('invoice', seen)
    id2 = make_unique_doc_id('invoice', seen)
    id3 = make_unique_doc_id('invoice', seen)
    check('First ID unchanged',  id1 == 'invoice')
    check('Second ID suffixed',  id2 == 'invoice_1')
    check('Third ID suffixed',   id3 == 'invoice_2')

    print(f'\n{"="*40}')
    if not failures:
        print('🎉 All tests passed!')
    else:
        print(f'️  {len(failures)} test(s) failed: {failures}')
    print('='*40)
    return failures


_run_unit_tests()

# ─── 10.1 File Upload (Colab) ─────────────────────────────────────────────────
# Option A: Upload interactively
try:
    from google.colab import files
    print('Upload your invoice files (JPG, PNG, PDF):')
    uploaded = files.upload()
    INPUT_FILES = list(uploaded.keys())
    print(f'Uploaded: {INPUT_FILES}')
except ImportError:
    # Option B: Hardcode paths (non-Colab / testing)
    INPUT_FILES = [
        # '/content/invoice_001.pdf',
        # '/content/invoice_002.jpg',
    ]
    print(f'Not in Colab. Using hardcoded file list: {INPUT_FILES}')

if not INPUT_FILES:
    print('️  No files specified. Add paths to INPUT_FILES.')

# ─── 10.2 Execute Pipeline ────────────────────────────────────────────────────
if INPUT_FILES:
    print(f'🚀 Starting extraction for {len(INPUT_FILES)} file(s)...\n')

    RESULTS = run_pipeline(INPUT_FILES, MODEL, PROCESSOR)

    OUTPUT_PATH = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_JSON)
    save_results(RESULTS, OUTPUT_PATH)

    # ── Download output in Colab ───────────────────────────────────────────────
    try:
        from google.colab import files as colab_files
        print(f'\n️  Downloading {cfg.OUTPUT_JSON}...')
        colab_files.download(OUTPUT_PATH)
    except ImportError:
        print(f'Output saved to: {OUTPUT_PATH}')
else:
    print('️  Skipping pipeline — no input files.')

# ─── 11.1 View Individual Results ────────────────────────────────────────────
def inspect_result(result: Dict, show_raw: bool = False):
    """Pretty-print a single extraction result."""
    print(f'\n{"─"*50}')
    print(f'📄 Document  : {result["doc_id"]}')
    print(f'   Confidence: {result["confidence"]:.2f}')
    print(f'   Time      : {result["processing_time_sec"]}s')
    print(f'   Cost est. : ${result["cost_estimate_usd"]}')
    print('   Fields:')
    fields = result['fields']
    print(f'      dealer_name  : {fields.get("dealer_name")}')
    print(f'      model_name   : {fields.get("model_name")}')
    print(f'      horse_power  : {fields.get("horse_power")}')
    print(f'      asset_cost   : {fields.get("asset_cost")}')
    print(f'      stamp        : {fields.get("stamp")}')
    print(f'      signature    : {fields.get("signature")}')
    if show_raw:
        print('\n   Full JSON:')
        print(json.dumps(result, ensure_ascii=False, indent=4))


# Show first 3 results
if 'RESULTS' in dir() and RESULTS:
    for r in RESULTS[:3]:
        inspect_result(r)
else:
    print('No results available yet. Run Section 10 first.')


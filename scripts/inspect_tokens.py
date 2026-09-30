"""Inspect the raw token stream of a page that loops, to tune the loop guard."""

import sys

import mlx.core as mx
from mlx_vlm import load, prompt_utils
from mlx_vlm.utils import load_config
from mlx_vlm.generate import dispatch
from PIL import Image
import pypdfium2 as pdfium

MODEL = "mlx-community/PaddleOCR-VL-1.5-4bit"
PDF = sys.argv[1] if len(sys.argv) > 1 else "samples/sample.pdf"
PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAXPX = int(sys.argv[3]) if len(sys.argv) > 3 else 1605632

model, processor = load(MODEL)
config = load_config(MODEL)
processor.image_processor.min_pixels = 112896
processor.image_processor.max_pixels = MAXPX

doc = pdfium.PdfDocument(PDF)
img = doc[PAGE - 1].render(scale=200 / 72).to_pil().convert("RGB")
doc.close()
scale = (MAXPX / (img.width * img.height)) ** 0.5
if scale < 1:
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

prompt = prompt_utils.apply_chat_template(processor, config, "OCR:", num_images=1)
tokens = []
for res in dispatch.stream_generate(
    model, processor, prompt, [img], max_tokens=4096, temperature=0.0
):
    if res.token is not None and res.token_ids is None:
        tokens.append(int(res.token.item()) if hasattr(res.token, "item") else int(res.token))
    elif res.finish_reason is not None and res.token_ids:
        tokens = list(res.token_ids)

txt = processor.tokenizer.decode(tokens)
print("total tokens:", len(tokens))
print("chars:", len(txt))

# Where does the text restart? Find repeated 64-token blocks and print offsets.
B = 64
for i in range(0, len(tokens) - B, 4):
    blk = tokens[i : i + B]
    hits = [j for j in range(i + 1, len(tokens) - B) if tokens[j : j + B] == blk]
    if hits:
        print(f"block at {i} also at {hits[:4]}")
        for j in hits[:3]:
            print(f"   @{j}: {processor.tokenizer.decode(tokens[j:j+40])!r}")
        break
else:
    print("no 64-token repeat found")

print("--- full text ---")
print(txt[:400])
print("...")
print(txt[-300:])

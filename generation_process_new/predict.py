import torch
from PIL import Image
from transformers import AutoProcessor
from .model_config import DEVICE, embedding_model_name, ATT_LABELS
from .model import VisionAttrTransformer

from AutoGPTQ.auto_gptq import AutoGPTQForCausalLM


class AttributePredictor:
    def __init__(self, checkpoint_path: str):
        self.processor = AutoProcessor.from_pretrained(
            embedding_model_name, trust_remote_code=True
        )

        self.vision_model = AutoGPTQForCausalLM.from_quantized(
            embedding_model_name,
            torch_dtype=torch.bfloat16,
            device=DEVICE,
            trust_remote_code=True,
            disable_exllama=True,
            disable_exllamav2=True,
        )

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        cfg = ckpt["config"]
        self.transformer = VisionAttrTransformer(
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            attr_sizes=cfg["attr_sizes"],
            drop_path_rate=cfg.get("drop_path_rate", 0.2)
        ).to(DEVICE)
        self.transformer.load_state_dict(ckpt["model_state_dict"])
        self.transformer.eval()

    def predict(self, pil_img: Image.Image) -> dict:
        proc = self.processor(text=" ", images=[pil_img], return_tensors="pt")
        pixel_values = [[t.to(DEVICE) for t in sub] for sub in proc["pixel_values"]]
        data = {
            "pixel_values": pixel_values,
            "tgt_sizes": proc["tgt_sizes"],
            "input_ids": torch.tensor([[220]], device=DEVICE),
            "image_bound": [[[0, 1]]],
        }
        with torch.no_grad():
            _, vision_hs = self.vision_model.get_vllm_embedding(data)
            emb = vision_hs[0].to(dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = torch.ones((1, emb.shape[1]), dtype=torch.bool, device=DEVICE)
            out = self.transformer(emb, mask)

        return {
            attr: ATT_LABELS[attr][torch.argmax(logits, dim=1).item()]
            for attr, logits in out.items()
        }

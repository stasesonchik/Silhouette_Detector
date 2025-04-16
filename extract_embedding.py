### === extract_embeddings.py ===

import torch
import os
import json
from PIL import Image
from transformers import AutoProcessor
from auto_gptq import AutoGPTQForCausalLM
import logging

# === ЛОГИРОВАНИЕ ===
# Создание логгера
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Создание обработчика для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Создание обработчика для записи в файл
file_handler = logging.FileHandler('embedding_process_log.txt', mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Создание форматтера
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Добавление обработчиков к логгеру
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# === НАСТРОЙКИ ===
torch.manual_seed(100)
device = "cuda:0"

# === ЗАГРУЗКА МОДЕЛИ ===
logging.info("started model loading")
model = AutoGPTQForCausalLM.from_quantized(
    'openbmb/MiniCPM-o-2_6-int4',
    torch_dtype=torch.bfloat16,
    device=device,
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True,
)
processor = AutoProcessor.from_pretrained('openbmb/MiniCPM-o-2_6-int4', trust_remote_code=True)
logging.info("finished model loading")

# === ПАРАМЕТРЫ ===
image_dir = "C:/Users\MrNik\PycharmProjects\PythonProject\AttributeDataset_Dec23_2024\pics"
json_dir = "C:/Users\MrNik\PycharmProjects\PythonProject\AttributeDataset_Dec23_2024/ann"
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)

# === ОБРАБОТКА ВСЕХ ИЗОБРАЖЕНИЙ ===
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
logging.info("started embedding process")

for idx, image_name in enumerate(image_files):
    image_path = os.path.join(image_dir, image_name)
    json_path = os.path.join(json_dir, image_name.replace('.jpg', '.json').replace('.png', '.json'))

    try:
        image = Image.open(image_path).convert("RGB")
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        gender_raw = metadata.get("пол", "не определен").strip().lower()
        if gender_raw == "мужчина":
            gender_label = 2
        elif gender_raw == "женщина":
            gender_label = 1
        else:
            gender_label = 0

        # Препроцессинг изображения
        processed = processor(text=" ", images=[image], return_tensors="pt")
        pixel_values = [[tensor.to(device) for tensor in sublist] for sublist in processed["pixel_values"]]

        data = {
            "pixel_values": pixel_values,
            "tgt_sizes": processed["tgt_sizes"],
            "input_ids": torch.tensor([[220]], device=device),
            "image_bound": [[[0, 1]]],
        }

        # Извлечение эмбеддинга
        with torch.no_grad():
            _, vision_hidden_states = model.get_vllm_embedding(data)

        embedding = vision_hidden_states[0].mean(dim=(0, 1)).cpu()  # [3584]

        # Сохранение эмбеддинга и метки пола
        save_path = os.path.join(output_dir, image_name + ".pt")
        torch.save({"embedding": embedding, "label": gender_label}, save_path)

        if idx % 1000 == 0:
            logging.info(f"processed {idx}/{len(image_files)} pics")

    except Exception as e:
        logging.warning(f"cannot process {image_name}: {e}")

logging.info("embedding process finished")

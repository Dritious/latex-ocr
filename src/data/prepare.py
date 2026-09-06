import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
import albumentations as A
import cv2

# 1. Настройки
NUM_SAMPLES = 5000       
IMG_DIR = "/content/dataset_images"
CSV_PATH = "/content/data.csv"
TARGET_HEIGHT = 64 # Базовая высота для выравнивания кусков

os.makedirs(IMG_DIR, exist_ok=True)

augmentor = A.Compose([
    A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.5),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
])

# 2. Универсальный рендер текста (если картинки нет в датасете)
def render_text_to_image(text, height=TARGET_HEIGHT):
    """Рендерит строку в PIL Image подстраивая ширину"""
    font = ImageFont.load_default() # ImageFont.truetype("arial.ttf", 32)
    
    box = font.getbbox(text)
    w = box[2] - box[0] + 20
    h = max(height, box[3] - box[1] + 20)
    
    img = Image.new('RGB', (w, h), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, (h - (box[3]-box[1]))//2), text, fill='black', font=font)
    
    return resize_to_height(img, height)

def resize_to_height(img, target_h):
    """Пропорционально меняет размер PIL Image под заданную высоту"""
    ratio = target_h / float(img.size[1])
    new_w = int(float(img.size[0]) * ratio)
    return img.resize((max(1, new_w), target_h), Image.Resampling.LANCZOS)

def fetch_sample(iterator, text_col, img_col=None, is_latex=False, max_words=None):
    """
    Универсально вытягивает 1 семпл из любого HF-итератора.
    Если img_col не задан или картинки нет — рендерит текст.
    """
    row = next(iterator)
    
    text = str(row[text_col]).replace('\n', ' ').strip()
    
    if max_words:
        words = text.split()
        if len(words) > max_words:
            start = random.randint(0, len(words) - max_words)
            text = " ".join(words[start:start + max_words])

    if img_col and img_col in row and row[img_col] is not None:
        img = row[img_col].convert('RGB')
        img = resize_to_height(img, TARGET_HEIGHT)
    else:
        img = render_text_to_image(text, TARGET_HEIGHT)

    if is_latex:
        text = f"${text}$"
        
    return text, img

def concat_images(images, direction='horizontal', padding=15):
    """Склеивает список PIL изображений в нужном направлении"""
    widths, heights = zip(*(i.size for i in images))
    
    if direction == 'horizontal':
        total_width = sum(widths) + (len(images) - 1) * padding
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height), color='white')
        offset = 0
        for im in images:
            y = (max_height - im.size[1]) // 2
            new_im.paste(im, (offset, y))
            offset += im.size[0] + padding
            
    elif direction == 'vertical':
        max_width = max(widths)
        total_height = sum(heights) + (len(images) - 1) * padding
        new_im = Image.new('RGB', (max_width, total_height), color='white')
        offset = 0
        for im in images:
            x = (max_width - im.size[0]) // 2
            new_im.paste(im, (x, offset))
            offset += im.size[1] + padding
            
    return new_im

iter_text = iter(load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True))
iter_latex = iter(load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train", streaming=True))

img_paths = []
texts = []

# Описываем шаблоны: порядок элементов (t - текст, l - латех)
templates = [
    (['t'], 'horizontal'),           # Только текст
    (['l'], 'horizontal'),           # Только латех
    (['t', 'l'], 'horizontal'),      # Текст -> Латех (слева направо)
    (['l', 't'], 'horizontal'),      # Латех -> Текст
    (['t', 'l', 't'], 'horizontal'), # Текст -> Латех -> Текст
    (['t', 'l'], 'vertical')         # Текст сверху, Латех снизу
]

from src.data.load_hf_data import load_ru_latex_datasets

def prepare_dataset():
    iter_ru, iter_latex = load_ru_latex_datasets()

    for i in range(NUM_SAMPLES):
        components_keys, direction = random.choice(templates)
        
        current_imgs = []
        current_texts = []
        
        for key in components_keys:
            if key == 't':
                txt, img = fetch_sample(iter_ru, text_col='text', img_col=None, max_words=40)
            elif key == 'l':
                txt, img = fetch_sample(iter_latex, text_col='latex_formula', img_col='image', is_latex=True)
                
            current_texts.append(txt)
            current_imgs.append(img)
        
        final_img_pil = concat_images(current_imgs, direction=direction)
        separator = " \n " if direction == 'vertical' else " "
        final_text = separator.join(current_texts)
        
        image_np = np.array(final_img_pil)
        augmented = augmentor(image=image_np)['image']
        
        img_filename = f"sample_{i}.jpg"
        save_path = os.path.join(IMG_DIR, img_filename)
        Image.fromarray(augmented).save(save_path)
        
        img_paths.append(save_path)
        texts.append(final_text)
        
        if (i+1) % 500 == 0:
            print(f"Обраработано {i+1} / {NUM_SAMPLES}")

    pd.DataFrame({'img_path': img_paths, 'text': texts}).to_csv(CSV_PATH, index=False)

if __name__=="__main__":
    prepare_dataset()

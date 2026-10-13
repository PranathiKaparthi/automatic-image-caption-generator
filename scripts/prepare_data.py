"""Dataset preparation script for evaluation and benchmarking."""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing import apply_synthetic_blur


def create_synthetic_test_dataset(data_dir: str = "data"):
    """
    Creates a set of diverse, high-quality benchmark images with ground-truth captions.
    Includes indoor, outdoor, vehicle, animal, and object scenes.
    Also creates motion-blurred and defocus-blurred pairs.
    """
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = [
        {
            "id": "scene_01_park",
            "category": "outdoor",
            "draw_func": "park",
            "references": [
                "A green sunny park with tall trees under a clear blue sky.",
                "A grassy public park with lush green trees and blue sky.",
                "Sunlight over a vibrant grassy field with deciduous trees.",
            ],
        },
        {
            "id": "scene_02_car",
            "category": "vehicle",
            "draw_func": "car",
            "references": [
                "A red sports car parked on an asphalt road in front of a building.",
                "A bright red automobile stationary on a street.",
                "A red vehicle parked on the road outside.",
            ],
        },
        {
            "id": "scene_03_desk",
            "category": "indoor",
            "draw_func": "desk",
            "references": [
                "A modern workspace desk with an open laptop and a white coffee cup.",
                "A laptop computer sitting on a wooden desk next to a mug.",
                "An office desk setup featuring a laptop and hot drink.",
            ],
        },
        {
            "id": "scene_04_dog",
            "category": "animal",
            "draw_func": "dog",
            "references": [
                "A playful golden retriever sitting on the green grass outdoors.",
                "A golden dog resting in a garden lawn.",
                "A happy yellow retriever dog sitting on a grassy lawn.",
            ],
        },
        {
            "id": "scene_05_fruits",
            "category": "food",
            "draw_func": "fruits",
            "references": [
                "A ceramic bowl filled with fresh red apples and yellow bananas.",
                "A fruit bowl containing ripe apples and bananas on a table.",
                "Fresh colorful fruits including apples and bananas in a bowl.",
            ],
        },
    ]

    records = []

    for item in metadata:
        width, height = 400, 300
        img = Image.new("RGB", (width, height), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)

        func_name = item["draw_func"]
        if func_name == "park":
            # Sky & grass
            draw.rectangle([0, 0, width, 180], fill=(135, 206, 235))
            draw.rectangle([0, 180, width, height], fill=(34, 139, 34))
            # Sun
            draw.ellipse([30, 20, 90, 80], fill=(255, 215, 0))
            # Trees
            draw.rectangle([100, 140, 120, 220], fill=(139, 69, 19))
            draw.ellipse([70, 70, 150, 150], fill=(0, 100, 0))
            draw.rectangle([260, 150, 280, 230], fill=(139, 69, 19))
            draw.ellipse([230, 80, 310, 160], fill=(0, 100, 0))

        elif func_name == "car":
            # Road & background
            draw.rectangle([0, 0, width, 190], fill=(180, 190, 200))
            draw.rectangle([0, 190, width, height], fill=(50, 50, 50))
            # Red Car body
            draw.rectangle([80, 160, 320, 230], fill=(220, 20, 60))
            draw.polygon([(120, 160), (160, 110), (250, 110), (290, 160)], fill=(200, 15, 50))
            # Windows
            draw.polygon([(135, 155), (165, 120), (200, 120), (200, 155)], fill=(173, 216, 230))
            draw.polygon([(205, 155), (205, 120), (245, 120), (275, 155)], fill=(173, 216, 230))
            # Wheels
            draw.ellipse([110, 210, 160, 260], fill=(20, 20, 20))
            draw.ellipse([240, 210, 290, 260], fill=(20, 20, 20))

        elif func_name == "desk":
            # Wall & wooden desk
            draw.rectangle([0, 0, width, 150], fill=(230, 230, 235))
            draw.rectangle([0, 150, width, height], fill=(180, 130, 80))
            # Laptop base & screen
            draw.rectangle([130, 120, 270, 200], fill=(60, 60, 65))
            draw.rectangle([140, 125, 260, 195], fill=(30, 144, 255))
            draw.rectangle([110, 200, 290, 220], fill=(150, 150, 155))
            # Coffee mug
            draw.rectangle([310, 170, 350, 215], fill=(250, 250, 250))
            draw.ellipse([305, 185, 320, 205], fill=(200, 200, 200))

        elif func_name == "dog":
            # Grass & sky
            draw.rectangle([0, 0, width, 140], fill=(176, 224, 230))
            draw.rectangle([0, 140, width, height], fill=(60, 179, 113))
            # Golden dog body
            draw.ellipse([140, 130, 260, 220], fill=(218, 165, 32))
            draw.ellipse([220, 90, 290, 160], fill=(218, 165, 32))  # Head
            draw.ellipse([215, 100, 235, 140], fill=(184, 134, 11))  # Ear
            draw.ellipse([270, 115, 285, 130], fill=(20, 20, 20))    # Nose
            # Legs
            draw.rectangle([150, 200, 170, 255], fill=(218, 165, 32))
            draw.rectangle([220, 200, 240, 255], fill=(218, 165, 32))

        elif func_name == "fruits":
            # Table & plate
            draw.rectangle([0, 0, width, 130], fill=(245, 245, 245))
            draw.rectangle([0, 130, width, height], fill=(160, 120, 90))
            draw.ellipse([100, 140, 300, 240], fill=(230, 230, 235))  # Bowl
            # Apples
            draw.ellipse([140, 130, 190, 180], fill=(220, 20, 60))
            draw.ellipse([200, 135, 250, 185], fill=(200, 30, 30))
            # Banana
            draw.arc([130, 150, 270, 210], start=20, end=160, fill=(255, 215, 0), width=15)

        # Save clean image
        clean_filename = f"{item['id']}_clean.png"
        clean_path = os.path.join(images_dir, clean_filename)
        img.save(clean_path, format="PNG")

        # Generate synthetic motion blur
        motion_img = apply_synthetic_blur(img, blur_type="motion", intensity=13)
        motion_filename = f"{item['id']}_motion_blur.png"
        motion_path = os.path.join(images_dir, motion_filename)
        motion_img.save(motion_path, format="PNG")

        # Generate synthetic defocus blur
        defocus_img = apply_synthetic_blur(img, blur_type="gaussian", intensity=11)
        defocus_filename = f"{item['id']}_defocus_blur.png"
        defocus_path = os.path.join(images_dir, defocus_filename)
        defocus_img.save(defocus_path, format="PNG")

        records.append({
            "id": item["id"],
            "category": item["category"],
            "clean_image_path": clean_path,
            "motion_blur_path": motion_path,
            "defocus_blur_path": defocus_path,
            "references": item["references"],
        })

    index_file = os.path.join(data_dir, "dataset_index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Dataset generated successfully with {len(records)} test scenes at '{data_dir}'.")
    return index_file


if __name__ == "__main__":
    create_synthetic_test_dataset()

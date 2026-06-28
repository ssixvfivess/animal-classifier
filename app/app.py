from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model_utils import load_checkpoint, predict_image  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "animal_group_classifier.pt"

st.set_page_config(page_title="Классификатор животных", page_icon="🐾", layout="centered")

st.title("Классификатор группы животного по фотографии")
st.write(
    "Загрузите фото животного. Модель определит, к какой группе оно относится: "
    "млекопитающее, птица, рыба, рептилия, амфибия, насекомое и т.д."
)

if not MODEL_PATH.exists():
    st.error("Модель пока не обучена. Сначала запустите обучение.")
    st.code("python src/train_model.py", language="bash")
    st.stop()

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, display_names, image_size = load_checkpoint(MODEL_PATH, device=device)
    return model, class_names, display_names, image_size, device

model, class_names, display_names, image_size, device = load_model()

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    if st.button("Определить группу животного"):
        result = predict_image(image, model, class_names, display_names, image_size=image_size, device=device)
        confidence_percent = result["confidence"] * 100

        st.subheader("Результат")
        st.success(f"{result['class_name']} — {confidence_percent:.1f}%")

        probabilities = pd.DataFrame(
            [
                {"Группа": class_name, "Вероятность": probability}
                for class_name, probability in result["probabilities"].items()
            ]
        ).sort_values("Вероятность", ascending=False)

        st.subheader("Вероятности по классам")
        st.dataframe(
            probabilities.assign(Вероятность=lambda df: (df["Вероятность"] * 100).round(2)),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(probabilities.set_index("Группа"))

st.divider()
st.caption(
    "Проект выполнен как учебная интеллектуальная система: обучение модели находится в src/train_model.py, "
    "интерфейс сайта — в app/app.py."
)

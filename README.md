# Итоговый проект: классификатор группы животного по фотографии

## Тема

Интеллектуальная система, которая принимает фотографию животного и определяет его группу.

Формально все животные относятся к царству Animalia, поэтому в интерфейсе используется более точная формулировка: **группа животного**.

Поддерживаемые группы:

- млекопитающее;
- птица;
- рыба;
- рептилия;
- насекомое.

## Что делает проект

1. Загружает изображения животных из папок по классам.
2. Обучает модель классификации изображений.
3. Сохраняет обученную модель в файл.
4. Запускает сайт на Streamlit.
5. Позволяет загрузить фото животного и получить предсказание.

## Стек

- Python
- PyTorch
- Torchvision
- ResNet18 Transfer Learning
- Scikit-learn
- Streamlit
- Pandas
- Matplotlib
- Pillow

## Структура проекта

```text
animal_kingdom_image_project/
├── app/
│   └── app.py
├── data/
│   ├── README.md
│   └── animal_photos/
│       ├── mammals/
│       ├── birds/
│       ├── fish/
│       ├── reptiles/
│       ├── amphibians/
│       ├── insects/
│       ├── arachnids/
│       └── crustaceans/
├── models/
├── src/
│   ├── model_utils.py
│   ├── predict.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Как подготовить данные

Фотографии нужно разложить по папкам внутри `data/animal_photos`.

Пример:

```text
data/animal_photos/mammals/cat_01.jpg
data/animal_photos/mammals/dog_01.jpg
data/animal_photos/birds/eagle_01.jpg
data/animal_photos/reptiles/snake_01.jpg
```

Названия папок — это классы, которые модель будет предсказывать.

## Установка зависимостей

В терминале из корня проекта:

```bash
pip install -r requirements.txt
```

## Обучение модели

После того как изображения разложены по папкам, запустите:

```bash
python3 src/train_model.py --freeze_backbone --epochs 10
```
<img width="506" height="270" alt="image" src="https://github.com/user-attachments/assets/424adc83-ff2a-4dc5-8c5d-f9be2cde5a64" />

## Запуск сайта

```bash
python3 -m streamlit run app/app.py
```

После запуска откроется сайт. На нём можно загрузить фото животного и получить результат классификации.

<img width="520" height="1191" alt="image" src="https://github.com/user-attachments/assets/5e0cf290-a054-49ed-87ce-bd372b104815" />


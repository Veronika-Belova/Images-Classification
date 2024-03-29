import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from model import MyResNet
import time

def about_project():
    st.markdown("<h1 style='text-align: center;'>Проект: классификация спортивных изображений</h1>", unsafe_allow_html=True)
    st.image('Снимок экрана 2024-03-29 в 19.10.49.png', use_column_width=True)


    st.header('Cодержание датасета')
    st.write('Коллекция спортивных изображений, охватывающих 100 различных видов спорта. Изображения имеют формат 224х224,3 jpg. Данные разделены на обучающие и тестовые каталоги.')
    st.write('Train_dataset - 13493 картинок')
    st.write('Valid_dataset - 500 картинок')
    
    st.header('Выбор модели')
    st.write('Была использована предобученная нейросеть ResNet18. Эта модель обучалась на датасете с изображениями, в котором 1000 классов.')
    st.write('Для корректного использования на наших данных был изменен полносвязный слой, количество выходных нейронов - 100')
    st.image('futureinternet-10-00080-g002.webp', use_column_width=True)


def prediction():
    def load_model():
        model = MyResNet()
        model.load_state_dict(torch.load('weights_for_model.pt', map_location=torch.device('cpu')))
        model.eval()
        return model

    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка классов
    idx2class = {0: 'air hockey',
    1: 'ampute football',
    2: 'archery',
    3: 'arm wrestling',
    4: 'axe throwing',
    5: 'balance beam',
    6: 'barell racing',
    7: 'baseball',
    8: 'basketball',
    9: 'baton twirling',
    10: 'bike polo',
    11: 'billiards',
    12: 'bmx',
    13: 'bobsled',
    14: 'bowling',
    15: 'boxing',
    16: 'bull riding',
    17: 'bungee jumping',
    18: 'canoe slamon',
    19: 'cheerleading',
    20: 'chuckwagon racing',
    21: 'cricket',
    22: 'croquet',
    23: 'curling',
    24: 'disc golf',
    25: 'fencing',
    26: 'field hockey',
    27: 'figure skating men',
    28: 'figure skating pairs',
    29: 'figure skating women',
    30: 'fly fishing',
    31: 'football',
    32: 'formula 1 racing',
    33: 'frisbee',
    34: 'gaga',
    35: 'giant slalom',
    36: 'golf',
    37: 'hammer throw',
    38: 'hang gliding',
    39: 'harness racing',
    40: 'high jump',
    41: 'hockey',
    42: 'horse jumping',
    43: 'horse racing',
    44: 'horseshoe pitching',
    45: 'hurdles',
    46: 'hydroplane racing',
    47: 'ice climbing',
    48: 'ice yachting',
    49: 'jai alai',
    50: 'javelin',
    51: 'jousting',
    52: 'judo',
    53: 'lacrosse',
    54: 'log rolling',
    55: 'luge',
    56: 'motorcycle racing',
    57: 'mushing',
    58: 'nascar racing',
    59: 'olympic wrestling',
    60: 'parallel bar',
    61: 'pole climbing',
    62: 'pole dancing',
    63: 'pole vault',
    64: 'polo',
    65: 'pommel horse',
    66: 'rings',
    67: 'rock climbing',
    68: 'roller derby',
    69: 'rollerblade racing',
    70: 'rowing',
    71: 'rugby',
    72: 'sailboat racing',
    73: 'shot put',
    74: 'shuffleboard',
    75: 'sidecar racing',
    76: 'ski jumping',
    77: 'sky surfing',
    78: 'skydiving',
    79: 'snow boarding',
    80: 'snowmobile racing',
    81: 'speed skating',
    82: 'steer wrestling',
    83: 'sumo wrestling',
    84: 'surfing',
    85: 'swimming',
    86: 'table tennis',
    87: 'tennis',
    88: 'track bicycle',
    89: 'trapeze',
    90: 'tug of war',
    91: 'ultimate',
    92: 'uneven bars',
    93: 'volleyball',
    94: 'water cycling',
    95: 'water polo',
    96: 'weightlifting',
    97: 'wheelchair basketball',
    98: 'wheelchair racing',
    99: 'wingsuit flying'}


    def preprocess_image(image):
        preprocess = transforms.Compose([
            transforms.Resize(244),
            transforms.ToTensor(),

        ])
        image = preprocess(image)
        return image

    def predict_image(image):
        image = image.convert('RGB')
        image_tensor = preprocess_image(image).unsqueeze(0).to(device)
        start_time = time.time()
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output).item()
        end_time = time.time()
        prediction_time = end_time - start_time
        return idx2class[predicted_class], round(prediction_time, 3)


    st.title("Kлассификация изображений")
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
            
        if st.button('Предсказать'):
            prediction = predict_image(image)
            st.write(f'Название класса: {prediction}')

        prediction = predict_image(image)

    def load_image_from_url(url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image

    image_url = st.text_input('Введите URL изображения:')
    if st.button('Загрузить и предсказать'):
        image = load_image_from_url(image_url)
        st.image(image, caption='Загруженное изображение', use_column_width=True)

        prediction, prediction_time = predict_image(image)
        st.write(f'Название класса: {prediction}')
        st.write(f'Время предсказания: {round(prediction_time, 3)} секунд')

def result():
    st.markdown("<h3 style='font-size: 24px;'>Обучение на 6 эпохах</h3>", unsafe_allow_html=True)
    st.image('Plt3.png', use_column_width=True)
    st.markdown("<h3 style='font-size: 24px;'>Метрики Accuracy, F1-score</h3>", unsafe_allow_html=True)

    data_list = [
    'train accuracy: 0.907',
    'valid accuracy: 0.881',
    'train F1score: 0.907',
    'valid F1score: 0.925'
]

    st.write(data_list)

navigation = st.sidebar.radio("Навигация", ['О проекте', 'Предсказание класса', 'Метрики'])

if navigation == 'О проекте':
    about_project()
elif navigation == 'Предсказание класса':
    prediction()
elif navigation == 'Метрики':
    result()
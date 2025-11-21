import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Определение маппингов и структуры (Без изменений) ---

# Mappings (от названия к числовому коду, который ждет модель)
PARKING_MAP = {
    'Подземная': 4, 'Многоуровневая': 3, 'Бесплатная во дворе': 2, 
    'Платная во дворе': 1, 'На крыше': 0
}

HOUSE_TYPE_MAP = {
    'Монолитно-кирпичный': 7, 'Монолитный': 6, 'Кирпичный': 4, 
    'Панельный': 2, 'Блочный': 1, 'Щитовой': 0, 
    'Деревянный': 3, 'Сталинский': 5, 'Старый фонд': 8
}

DISTRICT_MAP = {
    'ЦАО': 9, 'ВАО': 8, 'СЗАО': 7, 'СВАО': 6, 'ЗАО': 5, 
    'Районы': 4, 'ЮАО': 3, 'ЮВАО': 2, 'САО': 1
}

# 24 признака (без apartment_density)
MODEL_COLUMNS = [
    'rooms_count', 'ceiling_height_m', 'floors', 
    'bathroom_combined', 'bathroom_separate', 'balcony', 'loggia',
    'near_metro', 'parking_encoded', 'type_house_encoded', 
    'renovation_Дизайнерский', 'renovation_Евроремонт', 'renovation_Косметический', 
    'windows_На улицу', 'windows_На улицу и двор', 'garbage_chute_Нет', 
    'district_encoded', 'children_allowed_Нет', 'pets_allowed_Нет', 'premium_apartment', 
    'area_m2', 'metro_distance_min', 'cargo_elevator', 'passenger_elevator'
]

SCALER_FEATURES = ['area_m2', 'metro_distance_min', 'cargo_elevator', 'passenger_elevator']
NON_SCALER_FEATURES = [col for col in MODEL_COLUMNS if col not in SCALER_FEATURES]


# --- 2. Загрузка модели и скалера ---
try:
    model = joblib.load('realises/machine_learning/apartment_model.pkl')
    scaler = joblib.load('realises/machine_learning/scaler.pkl')
    
except FileNotFoundError:
    st.error("❌ **Ошибка загрузки:** Не найдены файлы 'apartment_model.pkl' или 'scaler.pkl'. Убедитесь, что они находятся в той же папке.")
    st.stop()
except Exception as e:
    st.error(f"❌ **Ошибка загрузки файлов:** {e}")
    st.stop()


# --- 3. Настройка страницы Streamlit ---
st.set_page_config(page_title="💰 Цена на квартиру", layout="wide", initial_sidebar_state="collapsed")

st.title("🏠 Предсказание стоимости жилья")
st.caption("Введите характеристики квартиры, чтобы получить прогноз цены в рублях.")
st.markdown("---")

# --- 4. Форма для ввода данных ---
with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)
    
    # === Колонка 1: Основные и Географические признаки ===
    with col1:
        
        # 🟢 Блок 1: Основные характеристики
        with st.expander("🛠️ Основные характеристики (Площадь, Комнаты)", expanded=True):
            
            c1, c2 = st.columns(2)
            with c1:
                rooms_count = st.number_input("Количество комнат", min_value=1, max_value=10, value=3, step=1)
                floors = st.number_input("Этаж", min_value=1, max_value=30, value=5, step=1)
                
            with c2:
                area_m2 = st.number_input("Общая площадь (м²)", min_value=10, max_value=500, value=65, step=1, help="Будет нормализовано.")
                # Единственное float поле
                ceiling_height_m = st.number_input("Высота потолков (м)", min_value=2.0, max_value=5.0, value=2.7, step=0.01)

        # 🗺️ Блок 2: География и Расположение
        with st.expander("🗺️ География и Инфраструктура", expanded=True):
            
            c1, c2 = st.columns(2)
            
            with c1:
                # District (Mapping from name to number)
                district_select = st.selectbox("Район", list(DISTRICT_MAP.keys()), index=4)
                
                # Near Metro (Binary)
                near_metro_bool = st.selectbox("Близость к метро", [1, 0], format_func=lambda x: "Рядом" if x == 1 else "Далеко")
                
            with c2:
                metro_distance_min = st.number_input("Расстояние до метро (мин)", min_value=1, max_value=60, value=15, step=1, help="Будет нормализовано.")
                
                # House Type (Mapping from name to number)
                house_type_select = st.selectbox("Тип дома", list(HOUSE_TYPE_MAP.keys()), index=6)

    # === Колонка 2: Удобства и Категории ===
    with col2:
        
        # 🚽 Блок 3: Санузлы и Балконы
        with st.expander("🚽 Санузлы и балконы", expanded=True):
            
            c1, c2 = st.columns(2)
            with c1:
                bathroom_combined = st.selectbox("Совмещенный санузел", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
                balcony = st.selectbox("Балкон", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
                
            with c2:
                bathroom_separate = st.selectbox("Раздельный санузел", [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")
                loggia = st.selectbox("Лоджия", [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")

        # ✨ Блок 4: Дополнительные опции
        with st.expander("✨ Ремонт, Лифты и Прочее", expanded=True):
            
            c1, c2 = st.columns(2)
            with c1:
                # Ремонт (One-Hot Logic)
                renovation_type = st.radio("Тип ремонта", ["Косметический", "Евроремонт", "Дизайнерский", "Нет"], index=3, horizontal=True)
                
                # Parking (Mapping from name to number)
                parking_select = st.selectbox("Тип парковки", list(PARKING_MAP.keys()), index=0)
                
                cargo_elevator = st.selectbox("Грузовой лифт", [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет", help="Будет нормализовано.")
                children_allowed = st.selectbox("Разрешено детям", [0, 1], format_func=lambda x: "Да" if x == 0 else "Нет")

            with c2:
                # Окна (One-Hot Logic)
                windows_type = st.radio("Куда выходят окна", ["На улицу", "На улицу и двор", "Во двор"], index=2, horizontal=True)

                premium_apartment_bool = st.selectbox("Премиум класс", [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")
                
                passenger_elevator = st.selectbox("Пассажирский лифт", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет", help="Будет нормализовано.")
                
                pets_allowed = st.selectbox("Разрешено животным", [0, 1], format_func=lambda x: "Да" if x == 0 else "Нет")
                garbage_chute = st.selectbox("Мусоропровод", [0, 1], format_func=lambda x: "Есть" if x == 0 else "Нет")
    
    st.markdown("---")
    submitted = st.form_submit_button("💰 СДЕЛАТЬ ПРЕДСКАЗАНИЕ", type="primary", use_container_width=True)
if submitted:
    
    # --- 5. Подготовка данных для модели (создание 24 признаков) ---
    
    # 5.1. Сбор и скалирование 4 признаков
    data_to_scale = pd.DataFrame([[
        area_m2, metro_distance_min, cargo_elevator, passenger_elevator
    ]], columns=SCALER_FEATURES)
    
    try:
        scaled_data = scaler.transform(data_to_scale)
        scaled_df = pd.DataFrame(scaled_data, columns=SCALER_FEATURES)
    except Exception as e:
        st.error(f"❌ **Критическая ошибка скалирования:** Скалер ожидает другое количество признаков (должно быть 4). Ошибка: {e}")
        st.stop()

    # 5.2. Сбор 20 ненормализованных признаков
    data_non_scaled = {
        'rooms_count': rooms_count, 'ceiling_height_m': ceiling_height_m, 'floors': floors, 
        'bathroom_combined': bathroom_combined, 'bathroom_separate': bathroom_separate, 'balcony': balcony, 'loggia': loggia,
        
        # One-Hot для ремонта
        'renovation_Дизайнерский': 1 if renovation_type == 'Дизайнерский' else 0,
        'renovation_Евроремонт': 1 if renovation_type == 'Евроремонт' else 0,
        'renovation_Косметический': 1 if renovation_type == 'Косметический' else 0,
        
        # One-Hot для окон
        'windows_На улицу': 1 if windows_type == 'На улицу' else 0,
        'windows_На улицу и двор': 1 if windows_type == 'На улицу и двор' else 0,
        
        # Остальные закодированные/бинарные признаки
        'near_metro': near_metro_bool,
        'garbage_chute_Нет': garbage_chute,
        'children_allowed_Нет': children_allowed,
        'pets_allowed_Нет': pets_allowed,
        'premium_apartment': premium_apartment_bool,
        
        # Mapped Encoded Features
        'parking_encoded': PARKING_MAP[parking_select],
        'type_house_encoded': HOUSE_TYPE_MAP[house_type_select],
        'district_encoded': DISTRICT_MAP[district_select]
    }
    
    non_scaled_df = pd.DataFrame([data_non_scaled])
    
    # 5.3. Объединение и выбор столбцов в ПРАВИЛЬНОМ ПОРЯДКЕ
    X_final_data = non_scaled_df.join(scaled_df)
    
    try:
        X_final = X_final_data[MODEL_COLUMNS]
        
        # --- 6. Предсказание ---
        prediction = model.predict(X_final)[0]
        
        # --- 7. Вывод результата ---
        st.markdown("---")
        st.balloons()
        
        formatted_price = f"{int(prediction):,}".replace(",", " ")
        
        st.success("## ✨ Прогноз готов! ✨")
        st.markdown(f"## **Предсказанная стоимость:**")
        st.markdown(f"## **{formatted_price} ₽**")
    except KeyError as e:
        st.error(f"❌ **Критическая ошибка:** В списке MODEL_COLUMNS неверное имя признака: {e}.")
    except Exception as e:
        st.error(f"❌ **Произошла непредвиденная ошибка:** {e}")
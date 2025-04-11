# app.py
import streamlit as st
import pandas as pd
from ast import literal_eval # Для безопасной обработки списков из строк
import os # Для проверки существования файла
import json # Для загрузки Карты Знаний из строки

# --- Конфигурация страницы ---
# st.set_page_config ДОЛЖНА быть первой командой Streamlit
st.set_page_config(
    layout="wide", # Используем широкую раскладку страницы
    page_title="Анализ дневника гимназиста v3.3", # Обновляем версию
    page_icon="📜" # Иконка во вкладке браузера
)

# --- Конфигурация путей ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Используем актуальные имена файлов
DATA_PATH_PKL = os.path.join(BASE_DIR, 'merged_diary_events_v3.pkl')
DATA_PATH_CSV = os.path.join(BASE_DIR, 'merged_diary_events_v3.csv')

# --- Карта Знаний (встроена для примера, лучше загружать из файла) ---
KNOWLEDGE_MAP_JSON = """
[
  {
    "category_id": "REV1848", "category_name": "Революции и конфликты 1848-1849 гг.",
    "subcategories": [
      {"subcategory_id": "REV1848_HUN", "subcategory_name": "Венгерское восстание 1848-1849 гг.", "events": [
          {"event_id": "REV1848_HUN_INT", "event_name": "Российская интервенция в Венгрии (1849)"},
          {"event_id": "REV1848_HUN_MIL", "event_name": "Военные действия (Венгрия, общее)"},
          {"event_id": "REV1848_HUN_CAP", "event_name": "Капитуляция венгерской армии (Вилагош)"},
          {"event_id": "REV1848_HUN_FIG", "event_name": "Упоминания личностей (Бем, Паскевич)"} ]},
      {"subcategory_id": "REV1848_ITA", "subcategory_name": "Революции в Италии", "events": [
          {"event_id": "REV1848_ITA_ROM", "event_name": "Римская республика (1849)"},
          {"event_id": "REV1848_ITA_AUS", "event_name": "Австро-сардинская война (1848-1849)"} ]},
      {"subcategory_id": "REV1848_GER", "subcategory_name": "Революции в Германии/Пруссии (общее)"},
      {"subcategory_id": "REV1848_FRA", "subcategory_name": "Революции во Франции (общее)"},
      {"subcategory_id": "REV1848_EUR", "subcategory_name": "Общеевропейский контекст/Беспорядки на Западе"}
    ]
  },
  {
    "category_id": "EPIDEMIC", "category_name": "Эпидемии и здоровье",
    "subcategories": [
       {"subcategory_id": "EPIDEMIC_CHOLERA", "subcategory_name": "Эпидемия холеры (1848-1849)", "events": [
            {"event_id": "EPIDEMIC_CHOLERA_SPB", "event_name": "Холера в Санкт-Петербурге"},
            {"event_id": "EPIDEMIC_CHOLERA_NOV", "event_name": "Холера в Новгородской губернии"},
            {"event_id": "EPIDEMIC_CHOLERA_VOL", "event_name": "Холера в Вологде (слухи/упоминания)"},
            {"event_id": "EPIDEMIC_CHOLERA_DTH", "event_name": "Смерть от холеры"},
            {"event_id": "EPIDEMIC_CHOLERA_FEAR", "event_name": "Опасения/Страх перед холерой"} ]},
       {"subcategory_id": "EPIDEMIC_OTH", "subcategory_name": "Другие болезни"}
    ]
  },
  {
    "category_id": "RU_POLITICS", "category_name": "Внутренняя политика и общество Российской Империи",
    "subcategories": [
      {"subcategory_id": "RU_POLITICS_REPRESS", "subcategory_name": "Репрессии и цензура (Николай I)", "events": [
            {"event_id": "RU_POLITICS_REPRESS_PETR", "event_name": "Дело петрашевцев"},
            {"event_id": "RU_POLITICS_REPRESS_CENS", "event_name": "Усиление цензуры печати"},
            {"event_id": "RU_POLITICS_REPRESS_SHAVE", "event_name": "Указ об обязательном бритье"},
            {"event_id": "RU_POLITICS_REPRESS_POLICE", "event_name": "Упоминания тайной полиции"} ]},
      {"subcategory_id": "RU_POLITICS_IMPERIAL", "subcategory_name": "Императорская семья и двор", "events": [
           {"event_id": "RU_POLITICS_IMPERIAL_CORON", "event_name": "Годовщина коронации Николая I"},
           {"event_id": "RU_POLITICS_IMPERIAL_DEATH", "event_name": "Смерть Великого Князя Михаила Павловича"} ]},
       {"subcategory_id": "RU_POLITICS_SOCIAL", "subcategory_name": "Социальные вопросы/настроения", "events": [
           {"event_id": "RU_POLITICS_SOCIAL_REVMOOD", "event_name": "Революционные настроения молодежи"} ]}
    ]
  },
   {
    "category_id": "RU_FOREIGN", "category_name": "Международные отношения (вне 1848)",
    "subcategories": [
      {"subcategory_id": "RU_FOREIGN_TURK", "subcategory_name": "Российско-турецкие отношения", "events": [
           {"event_id": "RU_FOREIGN_TURK_WAR", "event_name": "Слухи о возможной Русско-турецкой войне (1849)"},
           {"event_id": "RU_FOREIGN_TURK_DIPLO", "event_name": "Дипломатические миссии (Константинополь)"} ]}
      ]
   },
  {"category_id": "OTHER", "category_name": "Другое / Не классифицировано"}
]
"""
# Загрузка Карты Знаний
try:
    KNOWLEDGE_MAP = json.loads(KNOWLEDGE_MAP_JSON)
    KM_CATEGORY_NAMES = {cat['category_id']: cat['category_name'] for cat in KNOWLEDGE_MAP}
    KM_SUBCATEGORY_NAMES = { subcat['subcategory_id']: subcat['subcategory_name'] for cat in KNOWLEDGE_MAP if 'subcategories' in cat for subcat in cat['subcategories'] }
    KM_EVENT_NAMES = { event['event_id']: event['event_name'] for cat in KNOWLEDGE_MAP if 'subcategories' in cat for subcat in cat['subcategories'] if 'events' in subcat for event in subcat['events'] }
    print("Карта Знаний успешно загружена и обработана.")
except json.JSONDecodeError as e:
    st.error(f"Критическая ошибка: Не удалось загрузить Карту Знаний: {e}")
    KNOWLEDGE_MAP = []
    KM_CATEGORY_NAMES, KM_SUBCATEGORY_NAMES, KM_EVENT_NAMES = {}, {}, {}

# --- Функция загрузки данных с кэшированием ---
@st.cache_data
def load_data(pkl_path, csv_path):
    """Загружает и предварительно обрабатывает данные из Pickle или CSV."""
    df = None
    loaded_from = None
    if os.path.exists(pkl_path):
        try:
            df = pd.read_pickle(pkl_path)
            print(f"Данные успешно загружены из pickle: '{pkl_path}'.")
            loaded_from = 'pickle'
        except Exception as e:
            st.error(f"Ошибка при загрузке pickle файла '{pkl_path}': {e}. Попытка загрузить CSV.")
            df = None
    else:
        st.warning(f"Файл pickle '{pkl_path}' не найден.")

    if df is None:
        if os.path.exists(csv_path):
            st.warning(f"Попытка загрузить CSV: '{csv_path}'")
            try:
                # Укажем тип данных для entry_id при загрузке, если возможно
                dtype_spec = {'entry_id': 'Int64'} # Используем Int64 для поддержки NaN, если вдруг будут
                df = pd.read_csv(csv_path, dtype=dtype_spec)
                print("Данные успешно загружены из CSV.")
                loaded_from = 'csv'
                try:
                    df.to_pickle(pkl_path)
                    print(f"Данные сохранены в pickle формате: '{pkl_path}'.")
                except Exception as pkl_save_error:
                    print(f"Не удалось сохранить данные в pickle: {pkl_save_error}")
            except Exception as csv_load_error:
                st.error(f"Ошибка при загрузке CSV файла '{csv_path}': {csv_load_error}")
                return pd.DataFrame()
        else:
             st.error(f"Резервный CSV файл '{csv_path}' также не найден.")
             return pd.DataFrame()

    if df is not None and not df.empty:
        try:
            # === ИСПРАВЛЕНИЕ: Переименование колонки 'classification_id' в 'event_id' ===
            if 'classification_id' in df.columns and 'event_id' not in df.columns:
                df.rename(columns={'classification_id': 'event_id'}, inplace=True)
                print("Колонка 'classification_id' ПЕРЕИМЕНОВАНА в 'event_id'.")
            elif 'classification_id' in df.columns and 'event_id' in df.columns:
                 print("Предупреждение: Присутствуют обе колонки 'classification_id' и 'event_id'. Используется 'event_id'.")
                 # Опционально: удалить classification_id, если она точно не нужна
                 # df.drop(columns=['classification_id'], inplace=True)
            elif 'event_id' not in df.columns:
                 print("Ошибка: Отсутствует колонка 'event_id' или 'classification_id' для иерархии.")
                 # Создаем колонку event_id и заполняем N/A, чтобы избежать ошибок ниже
                 df['event_id'] = 'N/A'

            # 1. Даты
            date_col_to_use = None
            if 'source_date' in df.columns: date_col_to_use = 'source_date'
            elif 'date' in df.columns: date_col_to_use = 'date'; print("Info: Using 'date' for 'source_date_dt'.")
            if date_col_to_use: df['source_date_dt'] = pd.to_datetime(df[date_col_to_use], errors='coerce')
            else: df['source_date_dt'] = pd.NaT; print("Error: No date column found.")

            # 2. Ключевые слова
            if 'keywords' in df.columns and loaded_from == 'csv':
                 print("Обработка 'keywords' (из CSV)...")
                 def safe_literal_eval(val):
                    if isinstance(val, list): return val
                    try:
                        if pd.isna(val) or not isinstance(val, str): return []
                        if not val.strip() or val.strip() == '[]': return []
                        if val.startswith("'") and val.endswith("'"): val = val[1:-1]
                        if val.startswith('"') and val.endswith('"'): val = val[1:-1].replace('\\"', '"')
                        res = literal_eval(val)
                        return res if isinstance(res, list) else []
                    except: return []
                 df['keywords'] = df['keywords'].apply(safe_literal_eval)
            elif 'keywords' in df.columns:
                 def clean_keywords_list(lst):
                     if isinstance(lst, list): return [str(k).strip() for k in lst if k and isinstance(k, (str, int, float))]
                     return []
                 df['keywords'] = df['keywords'].apply(clean_keywords_list)

            # 3. Entry ID
            if 'entry_id' not in df.columns:
                print("Warning: Creating 'entry_id' from index.")
                df['entry_id'] = df.index
            df['entry_id'] = df['entry_id'].astype(int) # Убедимся, что тип int

            # 4. Заполнение NaN/пустых строк в event-колонках значением 'N/A'
            # Включаем 'event_id' в список, чтобы он тоже обработался
            event_cols = ['event_id', 'event_name', 'description', 'date_in_text', 'location',
                          'historical_context', 'confidence', 'classification_confidence', 'text_fragment']
            for col in event_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('N/A').replace([None, ''], 'N/A')
                else:
                    df[col] = 'N/A'
                    print(f"Warning: Added missing column '{col}' with 'N/A'.")

            # 5. Преобразуем event_id в строку ПОСЛЕ заполнения NaN и переименования
            if 'event_id' in df.columns:
                df['event_id'] = df['event_id'].astype(str)
                # Дополнительная очистка строк 'nan', 'None', которые могли появиться при конвертации
                df['event_id'] = df['event_id'].replace(['nan', 'None', 'null'], 'N/A', regex=False)

            print("Предварительная обработка данных завершена.")
            return df
        except Exception as e:
            st.error(f"Ошибка во время предварительной обработки данных: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()


# --- Вспомогательные функции для Карты Знаний (без изменений) ---
def get_categories():
    return [(cat['category_id'], cat['category_name']) for cat in KNOWLEDGE_MAP]

def get_subcategories(category_id):
    if not category_id: return []
    for cat in KNOWLEDGE_MAP:
        if cat['category_id'] == category_id and 'subcategories' in cat:
            return [(sub['subcategory_id'], sub['subcategory_name']) for sub in cat['subcategories']]
    return []

def get_events(subcategory_id):
    if not subcategory_id: return []
    for cat in KNOWLEDGE_MAP:
        if 'subcategories' in cat:
            for subcat in cat['subcategories']:
                if subcat['subcategory_id'] == subcategory_id and 'events' in subcat:
                    return [(ev['event_id'], ev['event_name']) for ev in subcat['events']]
    return []

def get_full_event_name(event_id):
    if not event_id or event_id == 'N/A' or event_id == 'OTHER' or pd.isna(event_id):
        return event_id
    event_id_str = str(event_id)
    if event_id_str in KM_EVENT_NAMES: return KM_EVENT_NAMES[event_id_str]
    if event_id_str in KM_SUBCATEGORY_NAMES: return KM_SUBCATEGORY_NAMES[event_id_str]
    if event_id_str in KM_CATEGORY_NAMES: return KM_CATEGORY_NAMES[event_id_str]
    return event_id_str

# --- Загрузка данных ---
df_merged = load_data(DATA_PATH_PKL, DATA_PATH_CSV)

# --- Основная часть приложения ---
st.title("Анализ исторических событий в дневнике гимназиста XIX века (v3.2)") # Обновляем версию

if df_merged.empty:
    st.warning("Не удалось загрузить данные.")
    st.stop()

# --- Отладочная информация (закомментирована) ---
# with st.expander("Отладочная информация о загруженных данных"):
#     st.write("Проверка загрузки данных:")
#     st.write(f"DataFrame пустой? {df_merged.empty}")
#     if not df_merged.empty:
#         st.write(f"Количество строк: {len(df_merged)}")
#         st.write(f"Колонки: {df_merged.columns.tolist()}")
#         st.subheader("Анализ колонки event_id (после переименования и очистки)")
#         if 'event_id' in df_merged.columns:
#             unique_event_ids = df_merged['event_id'].unique()
#             st.write("Уникальные значения в 'event_id':")
#             st.write(unique_event_ids)
#             st.write(f"Типы данных в 'event_id': {df_merged['event_id'].apply(type).value_counts()}")
#             st.write(f"Количество 'N/A' в 'event_id': {(df_merged['event_id'] == 'N/A').sum()}")
#             st.write(f"Количество 'OTHER' в 'event_id': {(df_merged['event_id'] == 'OTHER').sum()}")
#         else:
#             st.write("Колонка 'event_id' отсутствует.")
#         st.write("Первые 5 строк:")
#         st.dataframe(df_merged.head())

# --- Фильтры в боковой панели ---
st.sidebar.header("Фильтры отображения")

df_display = df_merged.copy()

# --- Иерархический фильтр по Карте Знаний ---
st.sidebar.subheader("Фильтр по категориям событий")
categories = get_categories()
category_options = [("ALL", "Все Категории")] + categories
selected_cat_tuple = st.sidebar.selectbox(
    "Категория:", options=category_options, format_func=lambda x: x[1], key='cat_filter'
)
selected_cat_id = selected_cat_tuple[0] if selected_cat_tuple else None

subcat_disabled = not (selected_cat_id and selected_cat_id != "ALL")
subcategory_options = [("ALL", "Все Подкатегории")]
if not subcat_disabled: subcategory_options.extend(get_subcategories(selected_cat_id))

selected_subcat_tuple = st.sidebar.selectbox(
    "Подкатегория:", options=subcategory_options, format_func=lambda x: x[1], key='subcat_filter', disabled=subcat_disabled
)
selected_subcat_id = selected_subcat_tuple[0] if selected_subcat_tuple else None

event_disabled = not (selected_subcat_id and selected_subcat_id != "ALL")
event_options = [("ALL", "Все События")]
if not event_disabled: event_options.extend(get_events(selected_subcat_id))

selected_event_tuple = st.sidebar.selectbox(
    "Событие:", options=event_options, format_func=lambda x: x[1], key='event_filter', disabled=event_disabled
)
selected_event_id = selected_event_tuple[0] if selected_event_tuple else None

# --- Применение иерархического фильтра ---
# Фильтруем только если выбрано что-то конкретное (не "ALL")
if selected_cat_id and selected_cat_id != "ALL":
    # Применяем фильтр на основе самого глубокого выбранного уровня
    filter_id = selected_cat_id
    if selected_subcat_id and selected_subcat_id != "ALL":
        filter_id = selected_subcat_id
        if selected_event_id and selected_event_id != "ALL":
            filter_id = selected_event_id

    def check_hierarchical_id(event_id_val):
        if not isinstance(event_id_val, str) or event_id_val == 'N/A':
            return False
        # Для последнего уровня (event_id) - точное совпадение
        if filter_id in KM_EVENT_NAMES:
            return event_id_val == filter_id
        # Для подкатегории и категории - совпадение или начало строки
        else:
            return event_id_val == filter_id or event_id_val.startswith(filter_id + '_')

    # Применяем фильтр
    # Важно: убедимся, что колонка event_id существует
    if 'event_id' in df_display.columns:
        df_display = df_display[df_display['event_id'].apply(check_hierarchical_id)]
    else:
         st.sidebar.error("Ошибка: Колонка 'event_id' не найдена для фильтрации.")


# --- Остальные фильтры ---
# (Код для остальных фильтров остается без изменений, он будет применен к df_display)
st.sidebar.subheader("Дополнительные фильтры")
# Дата
date_col_present = 'source_date_dt' in df_display.columns and not df_display['source_date_dt'].isna().all()
if date_col_present and not df_display.empty:
    min_date_val = df_display['source_date_dt'].min()
    max_date_val = df_display['source_date_dt'].max()
    if pd.notna(min_date_val) and pd.notna(max_date_val):
        min_date, max_date = min_date_val.date(), max_date_val.date()
        if min_date <= max_date:
            try:
                date_range = st.sidebar.date_input( "Диапазон дат записей:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key='date_filter' )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_display = df_display[ df_display['source_date_dt'].notna() & (df_display['source_date_dt'].dt.date >= start_date) & (df_display['source_date_dt'].dt.date <= end_date) ]
            except: pass

# Ключевые слова
keywords_col_present = 'keywords' in df_display.columns
if keywords_col_present and not df_display.empty:
    all_keywords_flat = [k for sublist in df_display['keywords'].dropna() if isinstance(sublist, list) for k in sublist if k and isinstance(k, str)]
    unique_keywords = sorted(list(set(all_keywords_flat)))
    if unique_keywords:
        selected_keywords = st.sidebar.multiselect("Ключевые слова (И):", options=unique_keywords, key='keyword_filter')
        if selected_keywords:
            def check_keywords(kw_list): return isinstance(kw_list, list) and all(k in kw_list for k in selected_keywords)
            df_display = df_display[df_display['keywords'].apply(check_keywords)]

# Уверенность (Идентификация)
confidence_col_present = 'confidence' in df_display.columns
if confidence_col_present and not df_display.empty:
    confidence_levels = sorted([lvl for lvl in df_display['confidence'].dropna().unique() if lvl != 'N/A'])
    if confidence_levels:
        selected_confidence = st.sidebar.multiselect("Уверенность (Идент.):", options=confidence_levels, key='confidence_filter')
        if selected_confidence: df_display = df_display[df_display['confidence'].isin(selected_confidence)]

# Уверенность (Классификация)
class_confidence_col_present = 'classification_confidence' in df_display.columns
if class_confidence_col_present and not df_display.empty:
    class_confidence_levels = sorted([lvl for lvl in df_display['classification_confidence'].dropna().unique() if lvl != 'N/A'])
    if class_confidence_levels:
        selected_class_confidence = st.sidebar.multiselect("Уверенность (Класс.):", options=class_confidence_levels, key='class_confidence_filter')
        if selected_class_confidence: df_display = df_display[df_display['classification_confidence'].isin(selected_class_confidence)]


# --- Отображение отфильтрованных данных ---
st.header("Обзор записей и событий (отфильтровано)")
display_columns = ['entry_id', 'date', 'event_id', 'event_name', 'description', 'location',
                   'confidence', 'classification_confidence', 'keywords']
actual_display_columns = [col for col in display_columns if col in df_display.columns]

if not df_display.empty:
    st.dataframe(df_display[actual_display_columns], height=300, use_container_width=True)
    st.write(f"Отобрано строк: {len(df_display)}")
else:
    st.info("Нет данных, соответствующих выбранным фильтрам.")


# --- Детальный просмотр записи ---
st.header("Детальный просмотр записи")

entry_id_present = 'entry_id' in df_display.columns
if not df_display.empty and entry_id_present:
    available_entry_ids = sorted(df_display['entry_id'].unique())
    if available_entry_ids:
        selected_entry_id = st.selectbox( "Выберите ID записи для просмотра:", options=available_entry_ids, key='entry_selector', index=None, placeholder="Выберите ID..." )
        if selected_entry_id is not None:
            entry_data_full = df_merged[df_merged['entry_id'] == selected_entry_id]
            if not entry_data_full.empty:
                diary_date = entry_data_full['date'].iloc[0] if 'date' in entry_data_full.columns else 'N/A'
                diary_text = entry_data_full['text'].iloc[0] if 'text' in entry_data_full.columns else 'Текст отсутствует'

                st.subheader(f"Запись от {diary_date} (ID: {selected_entry_id})")
                st.markdown("#### Полный текст записи:")
                st.markdown(f"<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; font-family: sans-serif; font-size: 14px;'>{diary_text}</div>", unsafe_allow_html=True)

                st.subheader("Извлеченные исторические события:")
                events_in_entry = entry_data_full[ (entry_data_full['event_name'] != 'N/A') ].drop_duplicates(subset=['event_name', 'text_fragment'])

                if not events_in_entry.empty:
                    for i, (_, event_row) in enumerate(events_in_entry.iterrows()):
                        st.markdown("---")
                        event_name_display = event_row.get('event_name', 'N/A')
                        # Используем event_id, т.к. переименовали колонку
                        event_id_display = event_row.get('event_id', 'N/A')
                        full_hierarchical_name = get_full_event_name(event_id_display)

                        st.markdown(f"**Событие {i+1}: {event_name_display}**")
                        with st.expander(f"Подробнее: {event_name_display} (ID: {event_id_display})"):
                            if full_hierarchical_name != event_id_display and full_hierarchical_name not in ['N/A', 'OTHER']:
                                 st.markdown(f"**Классификация:** {full_hierarchical_name} (`{event_id_display}`)")
                            else:
                                 st.markdown(f"**Классификация ID:** `{event_id_display}`")

                            st.markdown(f"**Уверенность (Класс.):** `{event_row.get('classification_confidence', 'N/A')}`")
                            st.markdown(f"**Уверенность (Идент.):** `{event_row.get('confidence', 'N/A')}`")
                            st.markdown(f"**Описание:** {event_row.get('description', 'N/A')}")
                            st.markdown(f"**Дата в тексте:** `{event_row.get('date_in_text', 'N/A')}`")
                            st.markdown(f"**Местоположение:** `{event_row.get('location', 'N/A')}`")

                            keywords = event_row.get('keywords', [])
                            if isinstance(keywords, list) and keywords:
                                valid_keywords = [str(k).strip() for k in keywords if k and isinstance(k, (str, int, float)) and str(k).strip()]
                                if valid_keywords: st.markdown(f"**Ключевые слова:** {' | '.join([f'`{k}`' for k in valid_keywords])}")
                            elif keywords != 'N/A': st.markdown(f"**Ключевые слова:** `{keywords}`")

                            st.markdown(f"**Исторический контекст:**"); st.caption(f"{event_row.get('historical_context', 'N/A')}")
                            st.markdown(f"**Цитируемый фрагмент текста:**"); st.info(f"{event_row.get('text_fragment', 'N/A')}")
                else:
                    st.info("Для этой записи не найдено извлеченных исторических событий.")
            else:
                 st.warning(f"Не удалось найти данные для записи с ID {selected_entry_id}.")
    else:
        st.info("Нет записей, соответствующих текущим фильтрам, для детального просмотра.")
else:
    st.info("Примените фильтры или загрузите данные для просмотра деталей записей.")
    if not entry_id_present: st.warning("Колонка 'entry_id' отсутствует в данных.")


# --- Визуализация: Временная шкала событий ---
st.header("Временная шкала событий")

timeline_possible = ('source_date_dt' in df_display.columns and
                     'event_name' in df_display.columns and # Проверяем event_name, т.к. event_id может быть N/A
                     not df_display.empty)

if timeline_possible:
    timeline_data = df_display[ (df_display['event_name'] != 'N/A') & df_display['source_date_dt'].notna() ].copy()
    if not timeline_data.empty:
        group_by_category = st.checkbox("Группировать по категориям на временной шкале", key="group_timeline")
        timeline_data.set_index('source_date_dt', inplace=True)
        aggregation_period = st.selectbox( "Период агрегации:", options=['D', 'W', 'M', 'Q', 'Y'], index=2, format_func=lambda x: {'D':'День', 'W':'Неделя', 'M':'Месяц', 'Q':'Квартал', 'Y':'Год'}.get(x, x), key='timeline_agg' )
        try:
            if group_by_category and 'event_id' in timeline_data.columns:
                 # Используем переименованную колонку event_id
                 timeline_data['category_id'] = timeline_data['event_id'].apply( lambda x: str(x).split('_')[0] if isinstance(x, str) and '_' in str(x) else ('OTHER' if str(x) == 'OTHER' else 'UNKNOWN') )
                 timeline_data['category_id'] = timeline_data['category_id'].replace('UNKNOWN', 'OTHER')
                 timeline_counts = timeline_data.groupby([ pd.Grouper(freq=aggregation_period), 'category_id' ]).size().unstack(fill_value=0)
                 category_names_map = KM_CATEGORY_NAMES.copy()
                 if 'OTHER' not in category_names_map: category_names_map['OTHER'] = 'Другое / Не классифицировано'
                 timeline_counts.rename(columns=category_names_map, inplace=True)
            else:
                 timeline_counts = timeline_data.resample(aggregation_period).size()
                 timeline_counts = timeline_counts.rename("Количество событий")
            if not timeline_counts.empty:
                st.line_chart(timeline_counts)
                grouping_text = " с группировкой по категориям" if group_by_category and 'category_id' in timeline_data.columns else ""
                st.caption(f"Количество извлеченных событий по периодам '{aggregation_period}'{grouping_text} (на основе даты записи).")
            else: st.info("Нет данных для построения временной шкалы после агрегации.")
        except Exception as e: st.error(f"Ошибка при построении временной шкалы: {e}"); st.exception(e)
    else: st.info("Нет событий с валидными датами в отфильтрованных данных для построения временной шкалы.")
else: st.info("Нет данных для построения временной шкалы.")

# --- Конец приложения ---
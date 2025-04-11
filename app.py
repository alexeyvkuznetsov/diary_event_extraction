# app.py
import streamlit as st
import pandas as pd
from ast import literal_eval # Для безопасной обработки списков из строк
import os # Для проверки существования файла

# --- Конфигурация страницы ---
# st.set_page_config ДОЛЖНА быть первой командой Streamlit
st.set_page_config(
    layout="wide", # Используем широкую раскладку страницы
    page_title="Анализ дневника гимназиста", # Название во вкладке браузера
    page_icon="📜" # Иконка во вкладке браузера
)

# --- Конфигурация путей ---
# Определяем путь к данным относительно текущего файла app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Предпочитаем pickle для скорости и сохранения типов данных
DATA_PATH_PKL = os.path.join(BASE_DIR, 'merged_diary_events.pkl')
# Резервный путь к CSV
DATA_PATH_CSV = os.path.join(BASE_DIR, 'merged_diary_events.csv')


# --- Функция загрузки данных с кэшированием ---
@st.cache_data # Кэшируем данные, чтобы не загружать их при каждом действии
def load_data(pkl_path, csv_path):
    """
    Загружает и предварительно обрабатывает данные из файла Pickle или CSV.

    Args:
        pkl_path (str): Путь к файлу .pkl.
        csv_path (str): Путь к файлу .csv (используется как резервный).

    Returns:
        pandas.DataFrame: Загруженный и обработанный DataFrame или пустой DataFrame в случае ошибки.
    """
    df = None
    loaded_from = None

    if os.path.exists(pkl_path):
        try:
            df = pd.read_pickle(pkl_path)
            print(f"Данные успешно загружены из pickle: '{pkl_path}'.")
            loaded_from = 'pickle'
        except Exception as e:
            st.error(f"Ошибка при загрузке pickle файла '{pkl_path}': {e}. Попытка загрузить CSV.")
            df = None # Сбрасываем df на случай ошибки
    else:
        st.warning(f"Файл pickle '{pkl_path}' не найден.")

    if df is None: # Если pickle не загружен или не найден
        if os.path.exists(csv_path):
            st.warning(f"Попытка загрузить CSV: '{csv_path}'")
            try:
                df = pd.read_csv(csv_path)
                print("Данные успешно загружены из CSV.")
                loaded_from = 'csv'
                # Попытка сохранить в pickle для будущих запусков
                try:
                    df.to_pickle(pkl_path)
                    print(f"Данные сохранены в pickle формате: '{pkl_path}' для ускорения будущих загрузок.")
                except Exception as pkl_save_error:
                    print(f"Не удалось сохранить данные в pickle: {pkl_save_error}")
            except Exception as csv_load_error:
                st.error(f"Ошибка при загрузке CSV файла '{csv_path}': {csv_load_error}")
                return pd.DataFrame()
        else:
             st.error(f"Резервный CSV файл '{csv_path}' также не найден.")
             return pd.DataFrame()

    # --- Очистка и преобразование данных (выполняется один раз при загрузке) ---
    if df is not None and not df.empty:
        try:
            # 1. Преобразование дат
            date_col_to_use = None
            if 'source_date' in df.columns:
                date_col_to_use = 'source_date'
            elif 'date' in df.columns:
                date_col_to_use = 'date'
                print("Предупреждение: Используется колонка 'date' дневника как основа для 'source_date_dt'.")

            if date_col_to_use:
                 # errors='coerce' превратит невалидные даты в NaT (Not a Time)
                 df['source_date_dt'] = pd.to_datetime(df[date_col_to_use], errors='coerce')
                 print(f"Колонка 'source_date_dt' создана из '{date_col_to_use}'.")
            else:
                print("Ошибка: Нет колонки 'source_date' или 'date'. Создаю пустую колонку 'source_date_dt'.")
                df['source_date_dt'] = pd.NaT # Not a Time для отсутствующих дат

            # 2. Преобразование 'keywords' из строки в список
            if 'keywords' in df.columns and loaded_from == 'csv': # Преобразуем только если грузили из CSV
                 print("Обработка колонки 'keywords' (загружено из CSV)...")
                 def safe_literal_eval(val):
                     if isinstance(val, list): return val
                     try:
                         if pd.isna(val) or not isinstance(val, str): return []
                         if not val.strip() or val.strip() == '[]': return []
                         # Убираем возможные лишние кавычки, если строка типа "'[...]'":
                         if val.startswith("'") and val.endswith("'"):
                             val = val[1:-1]
                         res = literal_eval(val)
                         return res if isinstance(res, list) else []
                     except (ValueError, SyntaxError, TypeError):
                         return []
                     except Exception as e:
                          print(f"Неожиданная ошибка при обработке keywords '{val}': {e}")
                          return []
                 df['keywords'] = df['keywords'].apply(safe_literal_eval)
            elif 'keywords' in df.columns:
                 # Проверка типов в keywords, если загружено из pickle (на всякий случай)
                  df['keywords'] = df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
                  print("Колонка 'keywords' проверена (загружено из pickle).")


            # 3. Убедимся, что entry_id существует
            if 'entry_id' not in df.columns:
                print("Предупреждение: Колонка 'entry_id' отсутствует. Создаю на основе индекса.")
                df['entry_id'] = df.index

            # 4. Обработка NaN в текстовых колонках события
            event_text_cols = ['event_name', 'description', 'date_in_text', 'location', 'historical_context', 'confidence', 'text_fragment']
            for col in event_text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('N/A') # Заполняем пропуски для отображения

            print("Предварительная обработка данных завершена.")
            return df

        except Exception as e:
            st.error(f"Ошибка во время предварительной обработки данных: {e}")
            return pd.DataFrame()
    else:
        # Если df пуст после загрузки
        return pd.DataFrame()


# --- Загрузка данных при запуске приложения ---
df_merged = load_data(DATA_PATH_PKL, DATA_PATH_CSV)

# --- Основная часть приложения ---
# Заголовок теперь идет ПОСЛЕ set_page_config
st.title("Анализ исторических событий в дневнике гимназиста XIX века")

# Проверка загрузки данных
if df_merged.empty:
    st.warning("Не удалось загрузить или обработать данные. Проверьте наличие файлов 'merged_diary_events.pkl' или 'merged_diary_events.csv' и их формат.")
    # st.stop() останавливает выполнение ВСЕХ последующих команд Streamlit в скрипте для текущей сессии.
    # Это полезно, чтобы предотвратить ошибки, если данные не загружены.
    st.stop()

# --- Фильтры в боковой панели ---
st.sidebar.header("Фильтры отображения")

# Создаем копию для фильтрации, чтобы не изменять исходный df_merged
df_filtered = df_merged.copy()

# --- Фильтр по дате ---
date_col_present = 'source_date_dt' in df_filtered.columns and not df_filtered['source_date_dt'].isna().all()
if date_col_present:
    min_date_val = df_filtered['source_date_dt'].min()
    max_date_val = df_filtered['source_date_dt'].max()
    # Проверка, что min/max не NaT
    if pd.notna(min_date_val) and pd.notna(max_date_val):
        min_date = min_date_val.date()
        max_date = max_date_val.date()
        if min_date <= max_date:
            try:
                date_range = st.sidebar.date_input(
                    "Выберите диапазон дат записей",
                    value=(min_date, max_date), # Устанавливаем начальные значения
                    min_value=min_date,
                    max_value=max_date,
                    key='date_filter' # Уникальный ключ для виджета
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Фильтруем строки, где дата события попадает в диапазон
                    df_filtered = df_filtered[
                        df_filtered['source_date_dt'].notna() &
                        (df_filtered['source_date_dt'].dt.date >= start_date) &
                        (df_filtered['source_date_dt'].dt.date <= end_date)
                    ]
            except Exception as e:
                st.sidebar.error(f"Ошибка фильтрации по дате: {e}")
        else:
            st.sidebar.warning("Некорректный диапазон дат (min > max).")
    else:
        st.sidebar.warning("Не удалось определить корректный диапазон дат для фильтрации (min/max - NaT).")
else:
    st.sidebar.info("Колонка с датами ('source_date') отсутствует или содержит некорректные значения.")


# --- Фильтр по ключевым словам ---
keywords_col_present = 'keywords' in df_filtered.columns
if keywords_col_present:
    all_keywords_flat = [k for sublist in df_filtered['keywords'].dropna() if isinstance(sublist, list) for k in sublist if k and isinstance(k, str)]
    unique_keywords = sorted(list(set(all_keywords_flat)))

    if unique_keywords:
        selected_keywords = st.sidebar.multiselect(
            "Фильтр по ключевым словам (И)",
            options=unique_keywords,
            key='keyword_filter'
        )
        if selected_keywords:
            def check_keywords(keyword_list):
                if not isinstance(keyword_list, list): return False
                return all(sel_k in keyword_list for sel_k in selected_keywords)
            df_filtered = df_filtered[df_filtered['keywords'].apply(check_keywords)]
    else:
        st.sidebar.info("Нет доступных ключевых слов для фильтрации в текущем наборе данных.")
else:
    st.sidebar.info("Колонка 'keywords' отсутствует.")


# --- Фильтр по уровню уверенности ---
confidence_col_present = 'confidence' in df_filtered.columns
if confidence_col_present:
    confidence_levels = sorted([lvl for lvl in df_filtered['confidence'].dropna().unique() if lvl != 'N/A'])
    if confidence_levels:
        selected_confidence = st.sidebar.multiselect(
            "Фильтр по уверенности",
            options=confidence_levels,
            key='confidence_filter'
        )
        if selected_confidence:
            df_filtered = df_filtered[df_filtered['confidence'].isin(selected_confidence)]
    else:
        st.sidebar.info("Нет доступных уровней уверенности для фильтрации.")
else:
    st.sidebar.info("Колонка 'confidence' отсутствует.")


# --- Отображение отфильтрованных данных ---
st.header("Обзор записей и событий (отфильтровано)")
display_columns = ['entry_id', 'date', 'event_name', 'description', 'location', 'confidence', 'keywords']
actual_display_columns = [col for col in display_columns if col in df_filtered.columns]

if not df_filtered.empty:
    # Показываем урезанную версию для обзора
    st.dataframe(df_filtered[actual_display_columns], height=300) # Ограничим высоту
    st.write(f"Отобрано строк: {len(df_filtered)}")
else:
    st.info("Нет данных, соответствующих выбранным фильтрам.")


# --- Детальный просмотр записи ---
st.header("Детальный просмотр записи")

entry_id_present = 'entry_id' in df_filtered.columns
if not df_filtered.empty and entry_id_present:
    available_entry_ids = sorted(df_filtered['entry_id'].unique())
    if available_entry_ids:
        selected_entry_id = st.selectbox(
            "Выберите ID записи для просмотра:",
            options=available_entry_ids,
            key='entry_selector',
            index=None, # Не выбирать ничего по умолчанию
            placeholder="Выберите ID..." # Подсказка
        )

        # Проверяем, что ID был выбран
        if selected_entry_id is not None:
            entry_data_full = df_merged[df_merged['entry_id'] == selected_entry_id]
            if not entry_data_full.empty:
                # Используем .iloc[0] для получения данных первой (и, возможно, единственной) строки с этим ID
                diary_date = entry_data_full['date'].iloc[0] if 'date' in entry_data_full.columns else 'N/A'
                diary_text = entry_data_full['text'].iloc[0] if 'text' in entry_data_full.columns else 'Текст отсутствует'

                st.subheader(f"Запись от {diary_date} (ID: {selected_entry_id})")
                # Используем markdown для возможности форматирования текста, если нужно
                st.markdown("#### Полный текст записи:")
                st.markdown(f"<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;'>{diary_text}</div>", unsafe_allow_html=True)
                # st.text_area("Полный текст записи:", diary_text, height=300, key=f'text_area_{selected_entry_id}') # Альтернатива

                st.subheader("Извлеченные исторические события:")
                events_in_entry = entry_data_full[
                    entry_data_full['event_name'].notna() & (entry_data_full['event_name'] != 'N/A')
                ].drop_duplicates(subset=['event_name', 'text_fragment'])

                if not events_in_entry.empty:
                    for i, (_, event_row) in enumerate(events_in_entry.iterrows()):
                        st.markdown("---")
                        event_name = event_row.get('event_name', 'N/A')
                        st.markdown(f"**Событие {i+1}: {event_name}**")
                        with st.expander(f"Подробнее о событии: {event_name}"):
                            st.markdown(f"**Описание:** {event_row.get('description', 'N/A')}")
                            st.markdown(f"**Дата в тексте:** {event_row.get('date_in_text', 'N/A')}")
                            st.markdown(f"**Местоположение:** {event_row.get('location', 'N/A')}")
                            st.markdown(f"**Уверенность:** {event_row.get('confidence', 'N/A')}")

                            keywords = event_row.get('keywords', [])
                            if isinstance(keywords, list) and keywords:
                                valid_keywords = [str(k).strip() for k in keywords if k and isinstance(k, str) and str(k).strip()]
                                if valid_keywords:
                                     st.markdown(f"**Ключевые слова:** {' | '.join([f'`{k}`' for k in valid_keywords])}")
                            elif keywords != 'N/A':
                                st.markdown(f"**Ключевые слова:** {keywords}")

                            st.markdown(f"**Исторический контекст:**")
                            st.caption(f"{event_row.get('historical_context', 'N/A')}")

                            st.markdown(f"**Цитируемый фрагмент текста:**")
                            st.info(f"{event_row.get('text_fragment', 'N/A')}")
                else:
                    st.info("Для этой записи не найдено извлеченных исторических событий.")
            else:
                 st.warning(f"Не удалось найти данные для записи с ID {selected_entry_id} в исходном DataFrame.")
    else:
        st.info("Нет записей, соответствующих текущим фильтрам, для детального просмотра.")
else:
    st.info("Примените фильтры или загрузите данные для просмотра деталей записей.")
    if not entry_id_present:
         st.warning("Колонка 'entry_id' отсутствует в данных.")


# --- Визуализация: Временная шкала событий ---
st.header("Временная шкала событий")

timeline_possible = ('source_date_dt' in df_filtered.columns and
                     'event_name' in df_filtered.columns and
                     not df_filtered.empty)

if timeline_possible:
    timeline_data = df_filtered[
        (df_filtered['event_name'] != 'N/A') &
        df_filtered['source_date_dt'].notna()
    ].copy()

    if not timeline_data.empty:
        timeline_data.set_index('source_date_dt', inplace=True)
        aggregation_period = st.selectbox(
            "Период агрегации для временной шкалы:",
            options=['D', 'W', 'M', 'Q', 'Y'],
            index=2,
            format_func=lambda x: {'D':'День', 'W':'Неделя', 'M':'Месяц', 'Q':'Квартал', 'Y':'Год'}.get(x, x),
            key='timeline_agg'
        )
        try:
            # Используем .size() для подсчета строк (событий) в каждом периоде
            timeline_counts = timeline_data.resample(aggregation_period).size()
            timeline_counts = timeline_counts.rename("Количество событий")

            if not timeline_counts.empty:
                st.line_chart(timeline_counts)
                st.caption(f"Количество извлеченных исторических событий по периодам '{aggregation_period}' (на основе даты записи в дневнике).")
            else:
                st.info("Нет данных для построения временной шкалы после агрегации по выбранному периоду.")
        except Exception as e:
            st.error(f"Ошибка при построении временной шкалы: {e}")
    else:
        st.info("Нет событий с валидными датами в отфильтрованных данных для построения временной шкалы.")
else:
    st.info("Необходимы колонки с датами ('source_date') и событиями ('event_name') для построения временной шкалы.")

# --- Конец приложения ---
# merge_script_v3.py
import pandas as pd
import json
import os

# --- Конфигурация ---
diary_csv_path = 'data.csv' # Должен содержать 'entry_id'
events_json_path = 'historical_events_v3.json' # Результат работы event_3.0.py
output_merged_csv_path = 'merged_diary_events_v3.csv'
output_merged_pickle_path = 'merged_diary_events_v3.pkl' # Pickle лучше сохраняет типы данных

print("--- Начало скрипта объединения данных (v3) ---")

# --- Проверка наличия входных файлов ---
if not os.path.exists(diary_csv_path):
    print(f"Ошибка: Файл дневника {diary_csv_path} не найден.")
    exit()
if not os.path.exists(events_json_path):
    print(f"Ошибка: Файл событий {events_json_path} не найден. Запустите скрипт извлечения (event_3.0.py) сначала.")
    exit()

# --- Загрузка данных ---
try:
    print(f"Загрузка данных дневника из {diary_csv_path}...")
    df_diary = pd.read_csv(diary_csv_path)
    # Проверка наличия entry_id в файле дневника
    if 'entry_id' not in df_diary.columns:
        print(f"Ошибка: Колонка 'entry_id' отсутствует в {diary_csv_path}. Запустите 'add_entry_id.py'.")
        exit()
    # Убедимся, что entry_id имеет подходящий тип (например, int)
    df_diary['entry_id'] = df_diary['entry_id'].astype(int)
    print(f"Загружено {len(df_diary)} записей дневника.")
    print(f"Колонки дневника: {df_diary.columns.tolist()}")


    print(f"Загрузка извлеченных событий из {events_json_path}...")
    with open(events_json_path, 'r', encoding='utf-8') as f:
        events_data = json.load(f)

    # Обработка случая, если JSON пуст или содержит не список
    if not events_data:
        print("Файл событий пуст. Событий для объединения нет.")
        df_events = pd.DataFrame() # Создаем пустой DataFrame событий
    elif isinstance(events_data, dict):
         print("Предупреждение: Файл событий содержит один объект, преобразую в список.")
         events_data = [events_data]
         df_events = pd.DataFrame(events_data)
    elif isinstance(events_data, list):
         df_events = pd.DataFrame(events_data)
    else:
        print(f"Ошибка: Неожиданный тип данных в {events_json_path}: {type(events_data)}. Ожидался список.")
        exit()

    if not df_events.empty:
        # Проверка наличия entry_id в файле событий
        if 'entry_id' not in df_events.columns:
            print(f"Ошибка: Колонка 'entry_id' отсутствует в {events_json_path}. Невозможно выполнить объединение.")
            exit()
        # Убедимся, что entry_id имеет подходящий тип (например, int)
        df_events['entry_id'] = df_events['entry_id'].astype(int)
        print(f"Загружено {len(df_events)} событий.")
        print(f"Колонки событий: {df_events.columns.tolist()}")
    else:
        # Если событий нет, df_merged будет просто копией df_diary с добавленными пустыми колонками событий
        pass


except FileNotFoundError as e:
    print(f"Ошибка при загрузке файла: {e}")
    exit()
except pd.errors.EmptyDataError:
     print(f"Ошибка: Файл {diary_csv_path} пуст.")
     exit()
except json.JSONDecodeError as e:
     print(f"Ошибка декодирования JSON из {events_json_path}: {e}")
     exit()
except KeyError as e:
     print(f"Ошибка: Отсутствует необходимая колонка {e} в одном из файлов.")
     exit()
except Exception as e:
    print(f"Неожиданная ошибка при загрузке данных: {e}")
    exit()

# --- Выполнение объединения (LEFT JOIN по entry_id) ---
print("Выполнение объединения данных по 'entry_id'...")

if not df_events.empty:
    # Определяем колонки, которые есть и в df_diary, и в df_events (кроме 'entry_id')
    # Это нужно, чтобы избежать дублирования информации, которую LLM могла вернуть (например, source_date)
    common_cols_to_drop = [col for col in df_events.columns if col in df_diary.columns and col != 'entry_id']
    if common_cols_to_drop:
        print(f"Удаление общих колонок из df_events перед слиянием (кроме entry_id): {common_cols_to_drop}")
        df_events_to_merge = df_events.drop(columns=common_cols_to_drop)
    else:
        df_events_to_merge = df_events

    # Объединяем df_diary (слева) с df_events_to_merge (справа)
    # Используем 'entry_id' как ключ
    try:
        df_merged = pd.merge(
            df_diary,
            df_events_to_merge,
            on='entry_id',       # Ключ для объединения
            how='left'          # 'left' сохраняет все записи из df_diary
        )
        print(f"Объединение завершено. Размер итогового DataFrame: {df_merged.shape}")
    except Exception as merge_error:
         print(f"Ошибка во время операции merge: {merge_error}")
         print("Создание DataFrame только с записями дневника.")
         df_merged = df_diary.copy()
         # Добавляем пустые колонки событий
         event_cols = ["event_id", "event_name", "description", "date_in_text", "source_date",
                       "location", "historical_context", "confidence", "classification_confidence",
                       "keywords", "text_fragment"]
         for col in event_cols:
             if col not in df_merged.columns:
                 df_merged[col] = pd.NA


else:
    # Если событий нет, создаем df_merged как копию df_diary и добавляем пустые колонки
    print("Событий для объединения нет. Создаем структуру с пустыми колонками событий.")
    df_merged = df_diary.copy()
    # Определяем ожидаемые колонки событий (на основе схемы v3)
    event_cols = ["event_id", "event_name", "description", "date_in_text", "source_date",
                  "location", "historical_context", "confidence", "classification_confidence",
                  "keywords", "text_fragment"]
    for col in event_cols:
        if col not in df_merged.columns:
            # Используем pd.NA для обозначения отсутствующих данных
            df_merged[col] = pd.NA
    print(f"Создан DataFrame с записями дневника. Размер: {df_merged.shape}")


# --- Анализ результатов объединения ---
# Проверяем наличие ключевой колонки события для статистики
event_key_col = 'event_name' # Или 'event_id', если он всегда должен быть при наличии события
if event_key_col in df_merged.columns:
    total_rows = len(df_merged)
    # Считаем строки, где событие было успешно извлечено (не NaN и не 'N/A')
    rows_with_events = df_merged[event_key_col].notna() & (df_merged[event_key_col] != 'N/A')
    num_rows_with_events = rows_with_events.sum()

    # Считаем количество уникальных записей дневника, для которых были извлечены события
    entries_with_events = df_merged.loc[rows_with_events, 'entry_id'].nunique()

    # Считаем количество уникальных записей дневника БЕЗ событий
    all_entry_ids = set(df_diary['entry_id'])
    entry_ids_with_events = set(df_merged.loc[rows_with_events, 'entry_id'])
    entries_without_events = len(all_entry_ids - entry_ids_with_events)

    original_diary_entries = len(df_diary)

    print(f"\nСтатистика объединения:")
    print(f" - Всего строк в итоговом DataFrame: {total_rows}")
    print(f" - Исходное количество записей дневника: {original_diary_entries}")
    print(f" - Количество уникальных записей дневника с извлеченными событиями: {entries_with_events}")
    print(f" - Количество уникальных записей дневника БЕЗ извлеченных событий: {entries_without_events}")
    print(f" - Общее количество строк, представляющих события (может быть > записей): {num_rows_with_events}")
    # Проверка: entries_with_events + entries_without_events должно быть равно original_diary_entries
    if entries_with_events + entries_without_events != original_diary_entries:
        print(f"Предупреждение: Сумма записей с/без событий ({entries_with_events + entries_without_events}) не совпадает с исходным количеством записей ({original_diary_entries}). Возможны проблемы с ID.")

else:
    print(f"\nСтатистика объединения: Ключевая колонка события '{event_key_col}' отсутствует в итоговом DataFrame.")


# --- Сохранение результата ---
try:
    print(f"Сохранение объединенного DataFrame в {output_merged_csv_path}...")
    # При сохранении в CSV списки в 'keywords' превратятся в строки.
    # Их нужно будет снова парсить при чтении CSV.
    df_merged.to_csv(output_merged_csv_path, index=False, encoding='utf-8')

    print(f"Сохранение объединенного DataFrame в {output_merged_pickle_path}...")
    # Pickle сохранит 'keywords' как списки.
    df_merged.to_pickle(output_merged_pickle_path)
    print("Результаты успешно сохранены.")
except Exception as e:
    print(f"Ошибка при сохранении объединенных данных: {e}")

print("\n--- Скрипт объединения данных (v3) завершен ---")

# Показать пример данных
print("\nПример объединенных данных (v3):")
print(df_merged.head())
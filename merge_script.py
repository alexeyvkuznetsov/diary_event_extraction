# merge_script.py
import pandas as pd
import json
import os

# --- Конфигурация ---
diary_csv_path = 'data.csv'
events_json_path = 'historical_events.json'
output_merged_csv_path = 'merged_diary_events.csv'
output_merged_pickle_path = 'merged_diary_events.pkl' # Pickle лучше сохраняет типы данных

print("--- Начало скрипта объединения данных ---")

# --- Проверка наличия входных файлов ---
if not os.path.exists(diary_csv_path):
    print(f"Ошибка: Файл дневника {diary_csv_path} не найден.")
    exit()
if not os.path.exists(events_json_path):
    print(f"Ошибка: Файл событий {events_json_path} не найден. Запустите скрипт извлечения (event_2.5.py) сначала.")
    exit()

# --- Загрузка данных ---
try:
    print(f"Загрузка данных дневника из {diary_csv_path}...")
    df_diary = pd.read_csv(diary_csv_path)
    # Добавляем уникальный ID записи на основе индекса - полезно для идентификации
    df_diary['entry_id'] = df_diary.index
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
except Exception as e:
    print(f"Неожиданная ошибка при загрузке данных: {e}")
    exit()

# --- Подготовка к объединению ---
# Стандартизация ключевых колонок (дат) для объединения
# Важно: преобразуем обе колонки дат к единому строковому формату 'YYYY-MM-DD'
# Это помогает избежать проблем со сравнением разных типов дат (Timestamp vs str)
try:
    if 'date' in df_diary.columns:
        df_diary['merge_date_key'] = pd.to_datetime(df_diary['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        print("Ошибка: Колонка 'date' отсутствует в файле дневника.")
        exit()

    if not df_events.empty:
        if 'source_date' in df_events.columns:
            df_events['merge_date_key'] = pd.to_datetime(df_events['source_date'], errors='coerce').dt.strftime('%Y-%m-%d')
             # Проверим, есть ли ключ для объединения в df_events
            if 'merge_date_key' not in df_events.columns:
                 print("Ошибка: Не удалось создать ключ 'merge_date_key' в данных событий.")
                 exit()
        else:
            print("Предупреждение: Колонка 'source_date' отсутствует в файле событий. Невозможно объединить события.")
            # Обнуляем df_events, чтобы дальше шло объединение без событий
            df_events = pd.DataFrame()


    # Проверяем наличие записей с некорректными датами (NaT после to_datetime)
    if df_diary['merge_date_key'].isna().any():
        print(f"Предупреждение: Обнаружены некорректные или отсутствующие даты в {diary_csv_path}. Эти записи могут не объединиться.")
    if not df_events.empty and df_events['merge_date_key'].isna().any():
        print(f"Предупреждение: Обнаружены некорректные или отсутствующие даты в {events_json_path}. Эти события могут не объединиться.")

except Exception as e:
    print(f"Ошибка при обработке дат для объединения: {e}")
    exit()


# --- Выполнение объединения (LEFT JOIN) ---
print("Выполнение объединения данных...")
if not df_events.empty and 'merge_date_key' in df_events.columns:
    # Объединяем df_diary (слева) с df_events (справа)
    df_merged = pd.merge(
        df_diary,
        df_events,
        on='merge_date_key', # Используем стандартизированный ключ
        how='left'          # 'left' сохраняет все записи из df_diary
    )
    print(f"Объединение завершено. Размер итогового DataFrame: {df_merged.shape}")
    # Удаляем временный ключ после объединения
    df_merged.drop(columns=['merge_date_key'], inplace=True)
else:
    # Если событий нет, создаем df_merged как копию df_diary и добавляем пустые колонки
    print("Событий для объединения нет. Создаем структуру с пустыми колонками событий.")
    df_merged = df_diary.copy()
    # Определяем ожидаемые колонки событий (на основе вашей JSON схемы)
    event_cols = ["event_name", "description", "date_in_text", "source_date",
                  "location", "historical_context", "confidence", "keywords", "text_fragment"]
    for col in event_cols:
        if col not in df_merged.columns:
            # Используем pd.NA для обозначения отсутствующих данных
            df_merged[col] = pd.NA
    # Удаляем временный ключ, если он был создан
    if 'merge_date_key' in df_merged.columns:
        df_merged.drop(columns=['merge_date_key'], inplace=True)
    print(f"Создан DataFrame с записями дневника. Размер: {df_merged.shape}")


# --- Анализ результатов объединения ---
if 'event_name' in df_merged.columns:
    total_rows = len(df_merged)
    rows_with_events = df_merged['event_name'].notna().sum()
    rows_without_events = df_merged['event_name'].isna().sum()
    original_diary_entries = len(df_diary)

    print(f"\nСтатистика объединения:")
    print(f" - Всего строк в итоговом DataFrame: {total_rows}")
    print(f" - Исходное количество записей дневника: {original_diary_entries}")
    print(f" - Строк, представляющих извлеченные события: {rows_with_events}")
    # Важно: rows_without_events - это строки из исходного дневника, для которых НЕ было найдено событий.
    # Их количество должно совпадать с кол-вом записей без событий.
    print(f" - Строк, соответствующих записям дневника БЕЗ событий: {rows_without_events}")
    # Проверка: rows_with_events + rows_without_events должно быть >= original_diary_entries
    # (Может быть больше, если одна запись дневника дала несколько событий)
else:
    print("\nСтатистика объединения: Колонка 'event_name' отсутствует в итоговом DataFrame.")


# --- Сохранение результата ---
try:
    print(f"Сохранение объединенного DataFrame в {output_merged_csv_path}...")
    df_merged.to_csv(output_merged_csv_path, index=False, encoding='utf-8')
    print(f"Сохранение объединенного DataFrame в {output_merged_pickle_path}...")
    df_merged.to_pickle(output_merged_pickle_path)
    print("Результаты успешно сохранены.")
except Exception as e:
    print(f"Ошибка при сохранении объединенных данных: {e}")

print("\n--- Скрипт объединения данных завершен ---")
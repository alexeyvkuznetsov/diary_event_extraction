import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time
import re # Импортирован re для использования внутри функции

# Загрузка переменных окружения из файла .env
load_dotenv()

# Настройка API клиента
client = OpenAI(api_key=os.getenv("FORGET_API_KEY"), base_url="https://forgetapi.ru/v1")

def extract_historical_events(text, diary_date):
    """
    Извлекает исторические события из текста дневника XIX века

    Args:
        text (str): Текст дневниковой записи
        diary_date (str): Дата дневниковой записи

    Returns:
        list: Список событий в формате JSON или пустой список, если события не найдены
               или произошла ошибка.
    """
    system_prompt = """Ты историк-аналитик, специализирующийся на истории России и Европы XIX века.
    Твоя задача - точно выявлять исторические события из текстов дневников того периода.
    Если в тексте нет упоминаний исторических событий, верни пустой массив []."""

    user_prompt = f"""
    Проанализируй следующий текст из дневника XIX века (запись от {diary_date}) и извлеки упоминания о исторических событиях.

    Для каждого найденного исторического события представь следующую информацию в формате JSON:
    {{
        "event_name": "краткое название события",
        "description": "описание события на основе текста",
        "date_in_text": "дата события, как она упоминается в тексте (если есть)",
        "source_date": "{diary_date}",
        "location": "место события (если указано в тексте)",
        "historical_context": "историческая значимость события",
        "confidence": "высокая/средняя/низкая (насколько уверенно определено событие)",
        "keywords": ["список", "ключевых", "слов", "связанных", "с событием"],
        "text_fragment": "точная цитата из текста, где упоминается событие"
    }}

    Ищи следующие типы исторических событий:
    1. Военные конфликты, революции, восстания
    2. Политические события (смена власти, реформы)
    3. Социальные явления (эпидемии, массовые миграции)
    4. Экономические события (кризисы, реформы)
    5. Культурные события (значимые для эпохи)

    Обрати особое внимание на:
    - Географические названия
    - Даты
    - Имена исторических личностей
    - Упоминания войн, конфликтов, революций
    - Упоминания болезней и эпидемий
    - Упоминания политических изменений

    Представь результат в виде массива JSON-объектов событий.
    Если событие малозначительно или недостаточно информации, не включай его.
    Если в тексте нет упоминаний исторических событий, верни пустой массив [].

    ТЕКСТ ДНЕВНИКА:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="gemini-1.5-pro-002",  # или другая модель
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3  # низкая температура для более точных и менее креативных ответов
        )

        content = response.choices[0].message.content

        # Попытка преобразовать ответ в JSON
        try:
            # Убираем возможные обрамляющие ```json ... ``` или ``` ... ```
            content_cleaned = re.sub(r'^```(json)?\s*|\s*```$', '', content, flags=re.IGNORECASE | re.DOTALL).strip()

            # Иногда модель возвращает пустую строку или объяснение вместо пустого массива
            if not content_cleaned or content_cleaned == '[]':
                 return []

            events = json.loads(content_cleaned)
            # Убедимся, что результат - это список
            if isinstance(events, list):
                return events
            else:
                print(f"Предупреждение: API вернул не список, а {type(events)}. Попытка обернуть в список.")
                print(f"Содержимое: {content_cleaned}")
                # Если это одиночный объект, оборачиваем его в список
                if isinstance(events, dict):
                     return [events]
                else:
                     print(f"Не удалось обработать ответ странного типа: {content_cleaned}")
                     return []

        except json.JSONDecodeError:
            # Если не удалось преобразовать, попробуем извлечь JSON из текста с помощью regex
            print(f"Предупреждение: Не удалось напрямую преобразовать ответ в JSON. Исходный ответ:\n---\n{content}\n---")
            # Ищем массив [...] или объект {...} (на всякий случай, если вернется один объект)
            json_match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
            if json_match:
                try:
                    extracted_json_str = json_match.group(0)
                    events = json.loads(extracted_json_str)
                    if isinstance(events, list):
                       return events
                    elif isinstance(events, dict):
                       print("Предупреждение: Извлечен одиночный JSON объект, обернут в список.")
                       return [events]
                    else:
                       print(f"Ошибка: Извлеченный JSON не является ни списком, ни объектом: {extracted_json_str}")
                       return []
                except json.JSONDecodeError:
                    print(f"Ошибка: Не удалось преобразовать даже извлеченный JSON: {extracted_json_str}")
                    return []
            else:
                print(f"Ошибка: Не удалось найти валидный JSON (массив или объект) в ответе.")
                return []

    except Exception as e:
        print(f"Ошибка при обращении к API или обработке ответа: {e}")
        # Пауза перед повторной попыткой, если есть ограничение на частоту запросов
        time.sleep(5) # Раскомментируйте, если нужно делать паузу при ошибках API
        return [] # Возвращаем пустой список в случае ошибки API

def process_diary_dataframe(df, output_file='historical_events.json'):
    """
    Обрабатывает датафрейм с дневниковыми записями и извлекает исторические события

    Args:
        df (pandas.DataFrame): Датафрейм с колонками 'text' и 'date'
        output_file (str): Путь к файлу для сохранения результатов

    Returns:
        list: Список всех найденных событий
    """
    all_events = []

    # Проверка наличия необходимых колонок перед началом обработки
    if 'text' not in df.columns or 'date' not in df.columns:
        print(f"Ошибка: В DataFrame отсутствуют необходимые колонки 'text' и/или 'date'. Обработка прервана.")
        return [] # Возвращаем пустой список, так как обработка невозможна

    total_entries = len(df)
    print(f"Начало обработки {total_entries} дневниковых записей...")

    for idx, row in df.iterrows():
        # Получаем текст и дату, обрабатываем возможные NaN значения
        text = row['text']
        date = row['date']

        # Проверка, что текст не пустой и является строкой
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            print(f"Пропуск записи {idx+1}/{total_entries}: пустой или некорректный текст.")
            continue

        # Проверка и форматирование даты (если необходимо)
        if pd.isna(date):
             print(f"Пропуск записи {idx+1}/{total_entries}: отсутствует дата.")
             continue
        # Преобразуем дату в строку, если она еще не строка (например, объект Timestamp)
        if not isinstance(date, str):
             try:
                 # Попытка стандартного форматирования, можно адаптировать под формат в CSV
                 date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
             except Exception as date_err:
                 print(f"Предупреждение в записи {idx+1}/{total_entries}: Не удалось стандартно отформатировать дату '{date}'. Используется как есть. Ошибка: {date_err}")
                 date_str = str(date) # Используем строковое представление как есть
        else:
             date_str = date # Дата уже является строкой

        print(f"Обработка записи {idx+1}/{total_entries} от {date_str}...")

        # Извлечение событий из текущей записи
        # Добавлена обработка пустых строк или NaN в тексте перед отправкой в API
        if text and isinstance(text, str) and text.strip():
             events = extract_historical_events(text, date_str)
        else:
             events = [] # Не отправляем пустой текст в API

        # Если найдены события, добавляем их в общий список
        if events and len(events) > 0:
            print(f"  Найдено {len(events)} событий в записи от {date_str}")
            # Добавляем каждое событие отдельно (если API вернул список)
            all_events.extend(events)
        else:
            # Сообщение выводится внутри extract_historical_events при ошибке или отсутствии событий
            # Можно добавить дополнительное логирование здесь при необходимости
            print(f"  В записи от {date_str} исторические события не найдены или произошла ошибка при извлечении.")

        # Пауза для соблюдения ограничений API (можно настроить)
        time.sleep(10) # Пауза в 1 секунду между запросами

    # Сохраняем все найденные события в JSON файл
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_events, f, ensure_ascii=False, indent=4)
        print(f"\nОбработка завершена. Найдено всего {len(all_events)} исторических событий.")
        print(f"Результаты сохранены в файл: {output_file}")
    except Exception as write_err:
        print(f"\nОшибка при сохранении результатов в файл {output_file}: {write_err}")
        print("Результаты НЕ сохранены.")


    return all_events

# ------------- Основная часть скрипта -------------
if __name__ == "__main__":
    csv_file_path = 'data.csv'  # Имя файла CSV
    output_json_file = 'historical_events.json' # Имя выходного JSON файла
    df = None # Инициализируем DataFrame как None

    print(f"Попытка загрузить данные из файла: {csv_file_path}")
    try:
        # Загружаем данные из CSV файла
        df = pd.read_csv(csv_file_path)

        # Проверка наличия необходимых колонок 'text' и 'date'
        if 'text' not in df.columns or 'date' not in df.columns:
            print(f"Ошибка: В файле {csv_file_path} отсутствуют необходимые колонки 'text' и/или 'date'.")
            df = None # Сбрасываем df, чтобы не продолжать обработку
        elif df.empty:
            print(f"Предупреждение: Файл {csv_file_path} успешно прочитан, но он пуст.")
            # Оставляем df пустым, обработка не начнется
        else:
            print(f"Данные успешно загружены из {csv_file_path}. Найдено записей: {len(df)}")
            # Опционально: Показать первые несколько строк для проверки
            # print("Первые 5 строк данных:")
            # print(df.head())

    except FileNotFoundError:
        print(f"Ошибка: Файл {csv_file_path} не найден. Убедитесь, что файл находится в той же директории, что и скрипт, или укажите полный путь.")
    except pd.errors.EmptyDataError:
         print(f"Ошибка: Файл {csv_file_path} пуст.")
    except Exception as e:
        print(f"Ошибка при чтении файла {csv_file_path}: {e}")

    # Продолжаем только если датафрейм успешно загружен, содержит нужные колонки и не пуст
    if df is not None and not df.empty:
        # Обрабатываем датафрейм
        events = process_diary_dataframe(df, output_file=output_json_file)

        # Вывод примеров результатов (если события были найдены)
        if events:
            print("\nПримеры найденных событий:")
            # Используем .get() для безопасного доступа к ключам словаря
            for i, event in enumerate(events[:3]): # Показываем до 3 примеров
                print(f"\nСобытие {i+1}:")
                print(f"  Название: {event.get('event_name', 'N/A')}")
                print(f"  Описание: {event.get('description', 'N/A')}")
                print(f"  Дата в тексте: {event.get('date_in_text', 'N/A')}")
                print(f"  Дата источника: {event.get('source_date', 'N/A')}")
                print(f"  Местоположение: {event.get('location', 'N/A')}")
        # Сообщение об отсутствии найденных событий уже выводится в конце process_diary_dataframe

    else:
        print("\nОбработка не была запущена из-за ошибки загрузки данных, отсутствия необходимых колонок или пустого файла.")
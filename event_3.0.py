# event_3.0.py
import os
import json
import pandas as pd
from openai import OpenAI # Используйте правильный импорт для вашего API
from dotenv import load_dotenv
from datetime import datetime
import time
import re

# Загрузка переменных окружения из файла .env
load_dotenv()

# Настройка API клиента
# Убедитесь, что используете правильный ключ и URL
client = OpenAI(api_key=os.getenv("FORGET_API_KEY"), base_url="https://forgetapi.ru/v1")

# --- Карта Знаний (Пример в виде JSON строки) ---
# В реальном приложении лучше загружать из отдельного файла
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

# Загружаем карту знаний один раз
try:
    KNOWLEDGE_MAP = json.loads(KNOWLEDGE_MAP_JSON)
except json.JSONDecodeError as e:
    print(f"Ошибка загрузки Карты Знаний: {e}")
    KNOWLEDGE_MAP = [] # Используем пустую карту в случае ошибки

def get_knowledge_map_text():
    """Преобразует Карту Знаний в текстовое представление для промпта."""
    text = "Используй следующую иерархию для классификации событий (Категория -> Подкатегория -> Событие ID):\n"
    for category in KNOWLEDGE_MAP:
        text += f"\nКатегория: {category['category_name']} (ID: {category['category_id']})\n"
        if "subcategories" in category:
            for subcat in category['subcategories']:
                 text += f"  Подкатегория: {subcat['subcategory_name']} (ID: {subcat['subcategory_id']})\n"
                 if "events" in subcat:
                     for event in subcat['events']:
                          text += f"    Событие: {event['event_name']} (ID: {event['event_id']})\n"
    text += "\nЕсли событие точно соответствует одному из перечисленных, используй его event_id. Если оно относится к подкатегории, но не соответствует конкретному событию, используй subcategory_id. Если оно относится только к категории, используй category_id. Если не подходит ни к чему, используй category_id='OTHER'."
    return text


def extract_historical_events(text, diary_date, entry_id): # Добавлен entry_id
    """
    Извлекает исторические события из текста дневника XIX века,
    используя entry_id и Карту Знаний.

    Args:
        text (str): Текст дневниковой записи
        diary_date (str): Дата дневниковой записи
        entry_id (int): Уникальный идентификатор записи

    Returns:
        list: Список событий в формате JSON или пустой список.
    """
    if not text or not isinstance(text, str) or not text.strip():
        print(f"Предупреждение: Пустой текст для записи ID {entry_id}. Пропуск.")
        return []

    knowledge_map_text = get_knowledge_map_text()

    system_prompt = """Ты историк-аналитик, специализирующийся на истории России и Европы XIX века.
Твоя задача - точно выявлять и классифицировать исторические события из текстов дневников того периода.
Используй предоставленную Карту Знаний для классификации событий.
Если в тексте нет упоминаний исторических событий, верни пустой массив []."""

    user_prompt = f"""
Проанализируй следующий текст из дневника XIX века (запись ID: {entry_id} от {diary_date}) и извлеки упоминания о исторических событиях.

Для каждого найденного исторического события представь следующую информацию в формате JSON:
{{
    "entry_id": {entry_id}, // ID исходной записи дневника
    "event_id": "ID_события_из_Карты_Знаний_или_NULL", // ID наиболее конкретного события, подкатегории или категории из Карты Знаний. Если ничего не подходит, используй "OTHER". Если классификация неуверенна, ставь null.
    "event_name": "краткое_название_события", // Используй название из Карты Знаний, если применимо, или сформулируй сам, если событие не в карте или ID="OTHER".
    "description": "описание события на основе текста",
    "date_in_text": "дата события, как она упоминается в тексте (если есть)",
    "source_date": "{diary_date}", // Дата записи дневника
    "location": "место события (если указано в тексте)",
    "historical_context": "историческая значимость события (кратко)",
    "confidence": "высокая/средняя/низкая", // Твоя уверенность в ИДЕНТИФИКАЦИИ события в тексте
    "classification_confidence": "высокая/средняя/низкая", // Твоя уверенность в ПРАВИЛЬНОСТИ классификации по Карте Знаний (поле event_id)
    "keywords": ["список", "ключевых", "слов"],
    "text_fragment": "точная цитата из текста, где упоминается событие"
}}

**Карта Знаний для классификации:**
{knowledge_map_text}

**Инструкции:**
1.  Ищи упоминания войн, революций, восстаний, эпидемий, смертей известных личностей, политических указов, международных отношений, значимых социальных явлений.
2.  Игнорируй чисто личные события автора (ссоры с товарищами, любовные переживания, если они не связаны напрямую с историческим контекстом), бытовые детали (погода, походы в церковь без связи с событием).
3.  Для поля "event_id" выбери наиболее точный ID из Карты Знаний (event_id > subcategory_id > category_id). Если событие историческое, но не подходит ни под одну категорию, используй "OTHER". Если не уверен в классификации, поставь null.
4.  Для поля "event_name": если использован ID из карты, возьми соответствующее event_name или subcategory_name. Если ID="OTHER" или null, сформулируй краткое название сам.
5.  Оценивай "confidence" (уверенность в наличии события в тексте) и "classification_confidence" (уверенность в правильности присвоенного ID из Карты Знаний) отдельно.
6.  Представь результат в виде массива JSON-объектов. Если событий нет, верни [].

ТЕКСТ ДНЕВНИКА (ЗАПИСЬ ID {entry_id}):
{text}
"""

    max_retries = 3
    retry_delay = 10 # Секунды
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # Проверьте актуальность модели
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Еще ниже для большей точности и следования инструкциям
                # response_format={"type": "json_object"} # Если API поддерживает принудительный JSON
            )

            content = response.choices[0].message.content

            # Улучшенная очистка и парсинг JSON
            content_cleaned = content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
            elif content_cleaned.startswith("```"):
                 content_cleaned = content_cleaned[3:]

            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]

            content_cleaned = content_cleaned.strip()

            if not content_cleaned or content_cleaned == '[]':
                 print(f"  Запись ID {entry_id}: События не найдены или LLM вернула пустой результат.")
                 return []

            # Попытка прямого парсинга
            try:
                events = json.loads(content_cleaned)
                if isinstance(events, list):
                     # Дополнительная проверка: убедимся, что entry_id в событиях соответствует запрошенному
                     for event in events:
                         if event.get("entry_id") != entry_id:
                              print(f"Предупреждение: Несоответствие entry_id в ответе LLM для записи {entry_id}. Найдено: {event.get('entry_id')}. Корректирую.")
                              event["entry_id"] = entry_id
                     return events
                elif isinstance(events, dict):
                     print(f"Предупреждение: API вернул один объект для ID {entry_id}, оборачиваю в список.")
                     if events.get("entry_id") != entry_id:
                          print(f"Предупреждение: Несоответствие entry_id в одиночном объекте LLM для записи {entry_id}. Найдено: {events.get('entry_id')}. Корректирую.")
                          events["entry_id"] = entry_id
                     return [events]
                else:
                     print(f"Ошибка: API вернул некорректный тип данных ({type(events)}) для ID {entry_id}. Ответ: {content_cleaned}")
                     return []
            except json.JSONDecodeError as e_direct:
                 print(f"Предупреждение: Не удалось напрямую распарсить JSON для ID {entry_id}. Ошибка: {e_direct}. Исходный ответ:\n---\n{content}\n---")
                 # Попытка извлечь JSON с помощью regex (ищем [...] или {...})
                 json_match = re.search(r'(\[.*?\]|\{.*?\})', content, re.DOTALL | re.IGNORECASE)
                 if json_match:
                     extracted_json_str = json_match.group(0)
                     try:
                         events = json.loads(extracted_json_str)
                         if isinstance(events, list):
                             for event in events:
                                 if event.get("entry_id") != entry_id:
                                     print(f"Предупреждение (regex): Несоответствие entry_id в ответе LLM для записи {entry_id}. Найдено: {event.get('entry_id')}. Корректирую.")
                                     event["entry_id"] = entry_id
                             return events
                         elif isinstance(events, dict):
                             print(f"Предупреждение (regex): Извлечен один объект для ID {entry_id}, оборачиваю в список.")
                             if events.get("entry_id") != entry_id:
                                 print(f"Предупреждение (regex): Несоответствие entry_id в одиночном объекте LLM для записи {entry_id}. Найдено: {events.get('entry_id')}. Корректирую.")
                                 events["entry_id"] = entry_id
                             return [events]
                         else:
                             print(f"Ошибка (regex): Извлеченный JSON не список/объект для ID {entry_id}. Извлечено: {extracted_json_str}")
                             return []
                     except json.JSONDecodeError as e_regex:
                         print(f"Ошибка (regex): Не удалось распарсить даже извлеченный JSON для ID {entry_id}. Ошибка: {e_regex}. Извлечено: {extracted_json_str}")
                         return []
                 else:
                     print(f"Ошибка: Не удалось найти валидный JSON в ответе для ID {entry_id}.")
                     return []

        except Exception as e:
            print(f"Ошибка при обращении к API или обработке ответа для ID {entry_id} (Попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Повторная попытка через {retry_delay} секунд...")
                time.sleep(retry_delay)
            else:
                print("Достигнуто максимальное количество попыток.")
                return [] # Возвращаем пустой список после всех неудачных попыток
    # Если цикл завершился без успешного return
    return []


def process_diary_dataframe(df, output_file='historical_events_v3.json'):
    """
    Обрабатывает датафрейм с дневниковыми записями (с entry_id)
    и извлекает исторические события.

    Args:
        df (pandas.DataFrame): Датафрейм с колонками 'entry_id', 'text', 'date'
        output_file (str): Путь к файлу для сохранения результатов

    Returns:
        list: Список всех найденных событий
    """
    all_events = []

    # Проверка наличия необходимых колонок
    required_cols = ['entry_id', 'text', 'date']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Ошибка: В DataFrame отсутствуют необходимые колонки: {missing_cols}. Обработка прервана.")
        return []

    total_entries = len(df)
    print(f"Начало обработки {total_entries} дневниковых записей...")

    start_time = time.time()

    for idx, row in df.iterrows():
        entry_id = row['entry_id']
        text = row['text']
        date_val = row['date']

        # Проверка текста
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            print(f"Пропуск записи ID {entry_id} ({idx+1}/{total_entries}): пустой или некорректный текст.")
            continue

        # Проверка и форматирование даты
        if pd.isna(date_val):
             print(f"Пропуск записи ID {entry_id} ({idx+1}/{total_entries}): отсутствует дата.")
             continue
        try:
            # Преобразуем дату в строку нужного формата
            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
        except Exception as date_err:
            print(f"Предупреждение в записи ID {entry_id} ({idx+1}/{total_entries}): Не удалось стандартно отформатировать дату '{date_val}'. Ошибка: {date_err}. Используется str().")
            date_str = str(date_val) # Используем строковое представление как есть


        print(f"Обработка записи ID {entry_id} ({idx+1}/{total_entries}) от {date_str}...")

        # Извлечение событий из текущей записи
        events = extract_historical_events(text, date_str, entry_id)

        if events: # Проверяем, что список не пустой и не None
            print(f"  Найдено {len(events)} событий в записи ID {entry_id}")
            all_events.extend(events)
        # else: # Сообщение об отсутствии событий или ошибке выводится внутри extract_historical_events
            # print(f"  В записи ID {entry_id} события не найдены или произошла ошибка.")

        # Пауза для соблюдения ограничений API
        time.sleep(15) # Увеличим паузу, т.к. запросы стали сложнее

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nОбработка {total_entries} записей завершена за {processing_time:.2f} секунд.")

    # Сохраняем все найденные события в JSON файл
    try:
        # Перед сохранением убедимся, что all_events это список
        if not isinstance(all_events, list):
            print(f"Ошибка: Результат 'all_events' не является списком ({type(all_events)}). Не могу сохранить.")
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_events, f, ensure_ascii=False, indent=4)
            print(f"Найдено всего {len(all_events)} исторических событий.")
            print(f"Результаты сохранены в файл: {output_file}")
    except Exception as write_err:
        print(f"\nОшибка при сохранении результатов в файл {output_file}: {write_err}")
        print("Результаты НЕ сохранены.")

    return all_events

# ------------- Основная часть скрипта -------------
if __name__ == "__main__":
    # Убедитесь, что data.csv УЖЕ содержит колонку 'entry_id'
    csv_file_path = 'data.csv'
    output_json_file = 'historical_events_v3.json' # Новое имя файла
    df = None

    print(f"Попытка загрузить данные из файла: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        required_cols = ['entry_id', 'text', 'date']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Ошибка: В файле {csv_file_path} отсутствуют колонки: {missing}.")
            print("Пожалуйста, сначала запустите скрипт 'add_entry_id.py'.")
            df = None
        elif df.empty:
            print(f"Предупреждение: Файл {csv_file_path} пуст.")
            # Оставляем df пустым
        else:
            print(f"Данные успешно загружены из {csv_file_path}. Найдено записей: {len(df)}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {csv_file_path} не найден.")
    except pd.errors.EmptyDataError:
         print(f"Ошибка: Файл {csv_file_path} пуст.")
    except Exception as e:
        print(f"Ошибка при чтении файла {csv_file_path}: {e}")

    # Продолжаем только если датафрейм успешно загружен и содержит нужные колонки
    if df is not None and not df.empty:
        events_list = process_diary_dataframe(df, output_file=output_json_file)

        # Вывод примеров результатов (если события были найдены)
        if events_list:
            print("\nПримеры найденных событий (v3):")
            for i, event in enumerate(events_list[:3]): # Показываем до 3 примеров
                print(f"\nСобытие {i+1}:")
                print(f"  Entry ID: {event.get('entry_id', 'N/A')}")
                print(f"  Event ID (Карта Знаний): {event.get('event_id', 'N/A')}")
                print(f"  Название: {event.get('event_name', 'N/A')}")
                print(f"  Уверенность (Идентификация): {event.get('confidence', 'N/A')}")
                print(f"  Уверенность (Классификация): {event.get('classification_confidence', 'N/A')}")
                print(f"  Описание: {event.get('description', 'N/A')[:100]}...") # Показываем начало описания
    else:
        print("\nОбработка не была запущена из-за ошибки загрузки данных или отсутствия необходимых колонок.")
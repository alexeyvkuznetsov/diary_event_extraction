import pandas as pd
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import time
import os
import random
import traceback
from dotenv import load_dotenv
import logging
import numpy as np # Для метрик качества
from collections import Counter # Для метрик качества

# -----------------------------------------------------------------------------
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# -----------------------------------------------------------------------------
load_dotenv()

# Пути к файлам

DATA_PATH = "data/diary_with_id.csv"
KNOWLEDGE_MAP_PATH = "knowledge_map.json" # Оставляем, так как create_extraction_prompt его использует
TEMP_DIR = "temp"
LAST_PROCESSED_FILE = "last_processed.txt"
TEMP_RESULTS_FILE = "results/revolution_events_temp.json"
FINAL_RESULTS_FILE = "results/revolution_events.json"

# Настройки API

MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"
#MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
#MODEL_NAME = "models/gemini-2.0-flash"

API_CALLS_PER_MINUTE = 7
MAX_RETRIES = 3
RETRY_WAIT_BASE = 20

SAFETY_SETTINGS = [
    {'category': HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HARASSMENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
]

api_calls_counter = 0
last_api_call_time = time.time()

# -----------------------------------------------------------------------------
# ЗАГРУЗКА И ФОРМАТИРОВАНИЕ КАРТЫ ЗНАНИЙ (остается как есть, т.к. используется в create_extraction_prompt)
# -----------------------------------------------------------------------------
def format_knowledge_node_for_prompt(node: Dict[str, Any], indent_level: int = 0) -> str:
    indent = "    " * indent_level
    name_prefix = "**" if indent_level == 0 else "*"
    name_suffix = "**" if indent_level == 0 else "*"
    id_str = f" (ID: {node['id']})" if 'id' in node else ""
    node_type_prefix = "Категория: " if indent_level == 0 else "Подкатегория: " if indent_level == 1 else ""

    line = f"{indent}{name_prefix}   {node_type_prefix}{node['name']}{id_str}{name_suffix}\n"

    if "children" in node and node["children"]:
        for child_node in node["children"]:
            line += format_knowledge_node_for_prompt(child_node, indent_level + 1)
    return line

def load_and_format_knowledge_map(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            structured_map = json.load(f)

        if not isinstance(structured_map, list):
            logger.error(f"Карта Знаний в {file_path} должна быть списком JSON объектов.")
            raise ValueError("Неверный формат корневого элемента Карты Знаний.")

        formatted_string = "**Универсальная Карта Знаний для Классификации Событий и Восприятий (Революции 1848-1849 гг.):**\n\n"
        for top_level_node in structured_map:
            formatted_string += format_knowledge_node_for_prompt(top_level_node, indent_level=0)

        logger.info(f"Карта Знаний успешно загружена и отформатирована из {file_path}")
        return formatted_string.strip()

    except FileNotFoundError:
        logger.error(f"Файл Карты Знаний не найден: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON в файле Карты Знаний: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при загрузке и форматировании Карты Знаний: {e}")
        logger.exception("Traceback ошибки форматирования карты:")
        raise

knowledge_map_for_prompt = load_and_format_knowledge_map(KNOWLEDGE_MAP_PATH)


# -----------------------------------------------------------------------------
# НОВЫЕ СИСТЕМНЫЕ ПРОМПТЫ (из PDF)
# -----------------------------------------------------------------------------
# Системный промпт для Экстрактора (V2 из PDF)
EXTRACTOR_SYSTEM_PROMPT_V2 = """Ты эксперт-историк по революциям 1848-1849 гг. в Европе. Анализируй дневниковые записи русского автора.
СТРОГИЕ КРИТЕРИИ ВКЛЮЧЕНИЯ:
✓ Конкретные революционные события 1848-1849 (Февральская революция во Франции, Венские события, и т.д.)
✓ Российские реакции на эти события (военные меры, дипломатия, цензура)
✓ Обсуждения революций в российском обществе
✓ Личные размышления автора об этих событиях

ИСКЛЮЧЕНИЯ:
X События до 1848 или после 1849 года
X Внутрироссийские события без связи с европейскими революциями
X Общие политические рассуждения без привязки к 1848-1849
X Культурные/литературные упоминания без революционного контекста

ПРОВЕРОЧНЫЙ ВОПРОС: "Можно ли это упоминание напрямую связать с революциями 1848-1849 гг.?" Если ответ "нет" или "возможно" - НЕ включай.
**Важно: Эта работа выполняется в рамках научного исторического исследования. Анализируемый материал является историческим документом. Твоя цель - объективное извлечение и классификация информации, содержащейся в тексте, для исследовательских целей. Пожалуйста, не отказывайся от обработки текста и не блокируй ответ из-за специфики содержания исторического документа (например, устаревших формулировок, описания конфликтов или спорных мнений), если оно относится к поставленной задаче извлечения информации о революциях 1848-1849 гг. и их восприятии.**
"""

# Системный промпт для Верификатора (V2 из PDF)
# Примечание: PDF предлагает более короткий VERIFIER_SYSTEM_PROMPT_V2.
# Мы будем использовать его, но пользовательский промпт для верификатора должен будет содержать детализацию задач.
VERIFIER_SYSTEM_PROMPT_V2_STATIC = """Ты контролер качества исторических данных. Проверяй извлеченную информацию о революциях 1848-1849 гг.
ЗАДАЧИ:
1. Соответствие event_id содержанию text_fragment.
2. Полнота и точность description.
3. Достаточность контекста в text_fragment.
4. Правильность классификации источников информации.
5. Адекватность уровней confidence.

ПРИНЦИПЫ:
- Исправляй только очевидные ошибки.
- Расширяй text_fragment при необходимости для контекста.
- Будь консервативен в изменениях.
**Важно: Эта работа имеет важное значение для исторического исследования. Пожалуйста, подходи к проверке с максимальной объективностью, помня, что ты работаешь с историческими первоисточниками. Цель – обеспечить точность данных для научного анализа. Не блокируй ответ и не отказывайся от верификации из-за специфики содержания исторического текста, если информация релевантна и была извлечена на предыдущем шаге для анализа революционных событий 1848-1849 гг. и их восприятия.**
"""

# -----------------------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ (PYDANTIC) - без изменений
# -----------------------------------------------------------------------------
class RevolutionEvent(BaseModel):
    entry_id: int = Field(..., description="Идентификатор записи дневника")
    event_id: Optional[str] = Field(None, description="Идентификатор события из Карты Знаний")
    event_name: str = Field("Неклассифицированное событие", description="Название события/аспекта, взятое из Карты Знаний или сформулированное для OTHER_1848/null.")
    event_subtype_custom: Optional[str] = Field(None, description="Более детальное кастомное название/описание типа события, если стандартный event_name по event_id слишком общий.")
    description: str = Field("Не указано", description="Описание события, основанное исключительно на тексте дневника")
    date_in_text: Optional[str] = Field(None, description="Дата события, указанная в тексте")
    source_date: str = Field(..., description="Дата записи в дневнике")
    location: str = Field("Не указано", description="Место события, как оно указано или подразумевается в тексте")
    location_normalized: Optional[str] = Field(None, description="Нормализованное основное место события для агрегации.")
    brief_context: str = Field("Не указано", description="Конкретный исторический факт (1-2 предложения), связанный с упоминанием.")
    information_source: str = Field("Не указан", description="Источник информации о событии для автора дневника")
    information_source_type: Optional[Literal[
        "Официальные источники (газеты, манифесты)",
        "Неофициальные сведения (слухи, разговоры в обществе)",
        "Личные наблюдения и опыт автора",
        "Информация от конкретного лица (именованный источник)",
        "Источник неясен/не указан"
    ]] = Field(None, description="Категоризированный тип источника информации автора.")
    confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уровень уверенности в корректности извлеченных данных")
    classification_confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уровень уверенности в правильности присвоенного event_id")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова из текстового фрагмента")
    text_fragment: str = Field("Не указано", description="Точный фрагмент текста (одно или несколько полных предложений).")


# -----------------------------------------------------------------------------
# УТИЛИТЫ ДЛЯ РАБОТЫ С API
# -----------------------------------------------------------------------------
def initialize_models():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    extractor_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=EXTRACTOR_SYSTEM_PROMPT_V2 # Используем новый системный промпт
    )
    verifier_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=VERIFIER_SYSTEM_PROMPT_V2_STATIC # Используем новый системный промпт
    )
    return extractor_model, verifier_model

# manage_api_rate_limit - без изменений

def manage_api_rate_limit():
    global api_calls_counter, last_api_call_time
    current_time = time.time()
    elapsed_time = current_time - last_api_call_time
    if elapsed_time >= 60:
        api_calls_counter = 0
        last_api_call_time = current_time
    if api_calls_counter >= API_CALLS_PER_MINUTE:
        wait_time = 60 - elapsed_time + random.uniform(3, 7)
        if wait_time > 0:
            logger.info(f"Достигнут лимит API. Ожидание {wait_time:.2f} секунд...")
            time.sleep(wait_time)
        api_calls_counter = 0
        last_api_call_time = time.time()
    api_calls_counter += 1
    time.sleep(random.uniform(0.5, 1.5))

# -----------------------------------------------------------------------------
# УТИЛИТЫ ДЛЯ ОБРАБОТКИ ДАННЫХ
# -----------------------------------------------------------------------------
# ensure_default_values - без изменений

def ensure_default_values(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event_dict, dict):
        logger.warning(f"ensure_default_values получил не словарь: {type(event_dict)}. Возвращаем как есть.")
        return event_dict

    event_dict["event_name"] = event_dict.get("event_name") or "Неклассифицированное событие"
    event_dict["description"] = event_dict.get("description") or "Не указано"
    event_dict["location"] = event_dict.get("location") or "Не указано"
    event_dict["brief_context"] = event_dict.get("brief_context") or "Не указано"
    event_dict["information_source"] = event_dict.get("information_source") or "Не указан"
    event_dict["confidence"] = event_dict.get("confidence") or "Medium"
    event_dict["classification_confidence"] = event_dict.get("classification_confidence") or "Medium"
    event_dict["text_fragment"] = event_dict.get("text_fragment") or "Не указано"
    event_dict["keywords"] = event_dict.get("keywords") or []

    optional_fields_to_check_for_empty_string = [
        "event_id", "date_in_text", "location_normalized",
        "information_source_type", "event_subtype_custom"
    ]
    for key in optional_fields_to_check_for_empty_string:
        if key in event_dict and event_dict[key] == "":
            event_dict[key] = None

    literal_fields_map = {
        "information_source_type": [
            "Официальные источники (газеты, манифесты)",
            "Неофициальные сведения (слухи, разговоры в обществе)",
            "Личные наблюдения и опыт автора",
            "Информация от конкретного лица (именованный источник)",
            "Источник неясен/не указан"
        ],
        "confidence": ["High", "Medium", "Low"],
        "classification_confidence": ["High", "Medium", "Low"]
    }
    for field, valid_values in literal_fields_map.items():
        current_value = event_dict.get(field)
        if current_value is not None and current_value not in valid_values:
            logger.warning(f"Недопустимое значение '{current_value}' для поля '{field}'. Устанавливаю в None.")
            event_dict[field] = None
    return event_dict

# load_diary_data - без изменений
def load_diary_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ПОЛЬЗОВАТЕЛЬСКОГО ПРОМПТА ЭКСТРАКТОРА (из PDF)
# -----------------------------------------------------------------------------
def create_extraction_user_prompt(entry_id: int, text: str, date: str, knowledge_map: str) -> str:
    # Адаптируем пример структуры JSON из PDF под полную Pydantic модель
    # и делаем его более общим, чтобы не вводить модель в заблуждение конкретными значениями.
    return f"""
Дневниковая запись от {date} (ID: {entry_id}):
"{text}"

Карта знаний:
{knowledge_map}

ЗАДАЧА: Найди ВСЕ упоминания революций 1848-1849 гг. и связанные с ними реакции/восприятия. Для каждого упоминания:
1. Определи `event_id` по Карте знаний (или OTHER_1848/null, если не подходит).
2. Извлеки ОДНО или НЕСКОЛЬКО ПОЛНЫХ предложений с достаточным контекстом для `text_fragment`.
3. Опиши событие/аспект своими словами на основе текста для `description`.
4. Определи источник информации для автора дневника для `information_source` и `information_source_type`.
5. Заполни остальные поля (`event_name`, `event_subtype_custom`, `date_in_text`, `location`, `location_normalized`, `brief_context`, `confidence`, `classification_confidence`, `keywords`) согласно их описанию и тексту.

ФОРМАТ: JSON массив объектов. Если ничего не найдено - пустой массив [].

Пример структуры ОДНОГО объекта в массиве (заполни все поля для каждого найденного события):
```json
{{
    "entry_id": {entry_id},
    "event_id": "ID_ИЗ_КАРТЫ_ЗНАНИЙ_ИЛИ_NULL",
    "event_name": "НАЗВАНИЕ_СОБЫТИЯ_ИЗ_КАРТЫ_ИЛИ_КАСТОМНОЕ",
    "event_subtype_custom": "УТОЧНЕНИЕ_ДЛЯ_ОБЩИХ_ID_ИЛИ_NULL",
    "description": "ОПИСАНИЕ_НА_ОСНОВЕ_ТЕКСТА",
    "date_in_text": "ДАТА_ИЗ_ТЕКСТА_ИЛИ_NULL",
    "source_date": "{date}",
    "location": "МЕСТО_ИЗ_ТЕКСТА_ИЛИ_НЕ_УКАЗАНО",
    "location_normalized": "НОРМАЛИЗОВАННОЕ_МЕСТО_ИЛИ_NULL",
    "brief_context": "КРАТКИЙ_ИСТОРИЧЕСКИЙ_ФАКТ_ИЛИ_НЕ_УКАЗАНО",
    "information_source": "ИСТОЧНИК_ИНФОРМАЦИИ_ДЛЯ_АВТОРА_ИЗ_ТЕКСТА",
    "information_source_type": "ТИП_ИСТОЧНИКА_ИЗ_СПИСКА_ИЛИ_NULL",
    "confidence": "High/Medium/Low",
    "classification_confidence": "High/Medium/Low",
    "keywords": ["КЛЮЧЕВОЕ_СЛОВО_1", "КЛЮЧЕВОЕ_СЛОВО_2"],
    "text_fragment": "ТОЧНАЯ_ЦИТАТА_С_КОНТЕКСТОМ"
}}
```
"""

# -----------------------------------------------------------------------------
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ВЕРИФИКАЦИИ ДАННЫХ (с новыми промптами)
# -----------------------------------------------------------------------------
def extract_revolution_events(entry_id: int, text: str, date: str, extractor_model, current_knowledge_map_str: str) -> List[Dict[str, Any]]:
    manage_api_rate_limit()
    # Используем новую функцию для создания пользовательского промпта
    user_prompt = create_extraction_user_prompt(entry_id, text, date, current_knowledge_map_str)

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = extractor_model.generate_content( # Модель extractor_model имеет system_instruction (EXTRACTOR_SYSTEM_PROMPT_V2)
                user_prompt,                             # Передаем динамический пользовательский промпт
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.5, # PDF не указывает температуру для экстрактора, ставим среднюю
                    response_mime_type="application/json"
                )
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                logger.error(f"Запись {entry_id}: Промпт экстрактора заблокирован, {error_message_detail}")
                raise ValueError(f"Blocked prompt for entry_id {entry_id}")

            if not response.candidates:
                logger.error(f"Запись {entry_id}: Нет кандидатов в ответе экстрактора.")
                raise ValueError(f"No candidates in response for entry_id {entry_id}")

            json_str = response.text
            events = json.loads(json_str)
            if not isinstance(events, list):
                logger.error(f"Экстрактор вернул не список для {entry_id}. Тип: {type(events)}. Ответ: {events}")
                raise TypeError("LLM returned non-list for event extraction")
            return events
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON от экстрактора для {entry_id}: {e}. Строка: {json_str}")
            if retry_count >= MAX_RETRIES -1: raise
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка экстракции для {entry_id} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE * retry_count)
            else:
                logger.error(f"Превышены попытки экстракции для {entry_id}. Пропускаем.")
                return []
    return []

def verify_event(event_data: Dict[str, Any], verifier_model, full_text: str, current_knowledge_map_str: str) -> Dict[str, Any]:
    manage_api_rate_limit()
    if not isinstance(event_data, dict):
        logger.error(f"verify_event получил не словарь: {type(event_data)}.")
        return event_data

    event_data_to_verify = ensure_default_values(event_data.copy())
    original_event_id = event_data_to_verify.get('event_id')
    original_text_fragment = event_data_to_verify.get('text_fragment')

    # Пользовательский промпт для верификатора, включающий все необходимые данные и инструкции
    user_prompt_for_verifier = f"""
    Проверь и при необходимости исправь следующую информацию о событии, извлеченную из дневника:
    ```json
    {json.dumps(event_data_to_verify, ensure_ascii=False, indent=2)}
    ```

    Полный текст дневниковой записи для контекста:
    "{full_text}"

    Используй следующую Карту Знаний для проверки классификации (`event_id` и `event_name`):
    {current_knowledge_map_str}

    Выполни следующие задачи по проверке и коррекции JSON-объекта (согласно твоей роли и предоставленной Карте Знаний, как указано в системной инструкции):
    1.  **Классификация и Уверенность:**
        *   Проверь `event_id`.
        *   Проверь `event_name`.
        *   Оцени и скорректируй `classification_confidence`.
    2.  **Содержание и Текстовая Основа:**
        *   Проверь `description`.
        *   Проверь `text_fragment` на полноту и контекст (должен содержать **полные** предложения).
        *   Проверь `keywords`.
    3.  **Атрибуты События:**
        *   Проверь `event_subtype_custom` (уточнение для общих ID или null).
        *   Проверь `location` и `location_normalized`.
        *   Проверь `date_in_text`.
    4.  **Источник Информации:**
        *   Проверь `information_source`.
        *   Проверь `information_source_type` (должен быть из **точного** списка).
    5.  **Контекст и Общая Уверенность:**
        *   Проверь `brief_context` (должен быть **конкретным историческим фактом**, не мнением).
        *   Оцени и скорректируй `confidence`.

    **Формат ответа:**
    Верни исправленную версию события ТОЛЬКО в формате JSON объекта. Не добавляй никакого текста до или после JSON.
    """

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = verifier_model.generate_content(
                user_prompt_for_verifier, # Модель verifier_model имеет system_instruction (VERIFIER_SYSTEM_PROMPT_V2_STATIC)
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.5, # PDF не указывает, можно оставить 0.5 или снизить для большей строгости
                    response_mime_type="application/json"
                )
            )
            # ... (остальная часть функции verify_event без изменений) ...
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                logger.error(f"Верификация для entry_id {event_data.get('entry_id')}: Промпт верификатора заблокирован, {error_message_detail}")
                raise ValueError("Blocked prompt for verification")

            if not response.candidates:
                logger.error(f"Верификация для entry_id {event_data.get('entry_id')}: Нет кандидатов в ответе верификатора.")
                raise ValueError("No candidates in response for verification")

            json_str = response.text
            verified_event = json.loads(json_str)
            if not isinstance(verified_event, dict):
                logger.error(f"Верификатор вернул не словарь для entry_id {event_data.get('entry_id')}. Тип: {type(verified_event)}. Ответ: {verified_event}")
                raise TypeError("Verifier returned non-dict")

            if verified_event.get('event_id') != original_event_id:
                logger.info(f"Верификатор изменил event_id с '{original_event_id}' на '{verified_event.get('event_id')}' для entry_id {event_data.get('entry_id')}")
            if verified_event.get('text_fragment') != original_text_fragment:
                logger.info(f"Верификатор изменил text_fragment для entry_id {event_data.get('entry_id')}")

            return ensure_default_values(verified_event)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON от верификатора для entry_id {event_data.get('entry_id')}: {e}. Строка: {json_str}")
            if retry_count >= MAX_RETRIES - 1: return event_data_to_verify
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка верификации для entry_id {event_data.get('entry_id')} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE * retry_count)
            else:
                logger.error(f"Превышены попытки верификации для entry_id {event_data.get('entry_id')}. Возвращаем исходное событие.")
                return event_data_to_verify
    return event_data_to_verify


# -----------------------------------------------------------------------------
# УЛУЧШЕННАЯ ВАЛИДАЦИЯ ПОЛЕЙ (из PDF, интегрируется в основной цикл)
# -----------------------------------------------------------------------------
def enhanced_field_validation(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Расширенная валидация полей с возможными исправлениями или логированием."""
    # Проверка text_fragment на достаточность контекста
    fragment = event_dict.get('text_fragment', '')
    if isinstance(fragment, str) and len(fragment.split()) < 5: # Минимум 5 слов для контекста
        # Не меняем confidence здесь, так как это задача верификатора, но логируем
        logger.warning(f"Entry ID {event_dict.get('entry_id')}: Короткий text_fragment: '{fragment}'")

    # Проверка логической связности (пример из PDF)
    description = event_dict.get('description', '')
    event_id = event_dict.get('event_id')
    if isinstance(description, str) and 'революц' in description.lower() and not event_id:
        logger.warning(f"Entry ID {event_dict.get('entry_id')}: Упоминание революции в description ('{description}') без event_id.")
        # Здесь можно было бы попытаться установить 'OTHER_1848', но лучше оставить для верификатора или ручного анализа

    # Можно добавить другие валидации, например, соответствие event_id и event_name (хотя это больше задача верификатора)

    return event_dict # Возвращаем event_dict, возможно, с изменениями (если бы они были)

# -----------------------------------------------------------------------------
# МЕТРИКИ КАЧЕСТВА (из PDF)
# -----------------------------------------------------------------------------
def quality_metrics(extracted_events: List[Dict[str, Any]], original_text_length: int) -> Dict[str, Any]:
    """Метрики качества извлечения."""
    if not extracted_events:
        return {
            'events_count': 0,
            'avg_fragment_length': 0,
            'confidence_distribution': Counter(),
            'classification_confidence_distribution': Counter(),
            'source_type_diversity': Counter(),
            'source_diversity': 0,
            'text_coverage': 0
        }

    fragment_lengths = [len(str(e.get('text_fragment', '')).split()) for e in extracted_events]

    metrics = {
        'events_count': len(extracted_events),
        'avg_fragment_length': np.mean(fragment_lengths) if fragment_lengths else 0,
        'confidence_distribution': Counter([e.get('confidence') for e in extracted_events]),
        'classification_confidence_distribution': Counter([e.get('classification_confidence') for e in extracted_events]),
        'source_type_diversity': Counter([e.get('information_source_type') for e in extracted_events]),
        'source_diversity': len(set(e.get('information_source') for e in extracted_events)),
        'text_coverage': sum(len(str(e.get('text_fragment', ''))) for e in extracted_events) / original_text_length if original_text_length > 0 else 0
    }
    return metrics


# -----------------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДНЕВНИКА
# -----------------------------------------------------------------------------
def process_diary():
    data = load_diary_data(DATA_PATH)
    if data.empty:
        logger.error("Не удалось загрузить данные дневника. Завершение работы.")
        return

    logger.info("Инициализация моделей...")
    try:
        extractor_model, verifier_model = initialize_models()
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации моделей: {e}. Завершение работы.")
        return

    all_extracted_events_for_metrics: List[Dict[str, Any]] = [] # Для сбора всех событий для итоговых метрик
    total_text_length_for_metrics = 0 # Для расчета text_coverage

    all_events: List[Dict[str, Any]] = []
    for dir_path in [TEMP_DIR, os.path.dirname(TEMP_RESULTS_FILE), os.path.dirname(FINAL_RESULTS_FILE)]:
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                logger.info(f"Создана директория {dir_path}")
            except OSError as e:
                logger.error(f"Не удалось создать директорию {dir_path}: {e}. Завершение работы.")
                return


    last_processed_id = 0
    if os.path.exists(LAST_PROCESSED_FILE):
        try:
            with open(LAST_PROCESSED_FILE, "r") as f:
                last_processed_id = int(f.read().strip())
            logger.info(f"Найдена информация о последней обработанной записи: {last_processed_id}")
        except (ValueError, IOError) as e:
            logger.warning(f"Не удалось прочитать ID из {LAST_PROCESSED_FILE}: {e}. Начинаем с начала.")

    if os.path.exists(TEMP_RESULTS_FILE):
        try:
            with open(TEMP_RESULTS_FILE, "r", encoding="utf-8") as f:
                all_events = json.load(f)
            logger.info(f"Загружено {len(all_events)} уже обработанных событий из {TEMP_RESULTS_FILE}.")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Не удалось загрузить события из {TEMP_RESULTS_FILE}: {e}. Начинаем с нуля.")
            all_events = []

    for index, row in data.iterrows():
        current_entry_id_raw = row.get('entry_id')
        current_date = row.get('date')
        current_text = row.get('text')

        if pd.isna(current_entry_id_raw) or pd.isna(current_date) or pd.isna(current_text):
            logger.warning(f"Пропуск строки DataFrame {index} из-за отсутствующих данных (entry_id, date или text).")
            continue

        current_entry_id = int(current_entry_id_raw)
        current_text = str(current_text)
        total_text_length_for_metrics += len(current_text)


        if current_entry_id <= last_processed_id:
            # Если мы пропускаем уже обработанные, нам все равно нужны их события для финальных метрик,
            # если они были в TEMP_RESULTS_FILE.
            # Однако, если TEMP_RESULTS_FILE не содержал их, или мы начинаем заново, то их не будет.
            # Это усложнение. Проще всего считать метрики только по вновь обработанным.
            # Либо, если all_events загружен, он уже содержит старые.
            # Для простоты, будем считать метрики по всем событиям в `all_events` в конце.
            continue

        logger.info(f"Обработка записи {current_entry_id} от {current_date}...")
        temp_file = os.path.join(TEMP_DIR, f"entry_{current_entry_id}.json")

        if os.path.exists(temp_file):
            logger.info(f"Запись {current_entry_id} уже обработана (temp-файл), загружаем...")
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    entry_events_from_temp = json.load(f)

                existing_event_keys = {(evt.get('entry_id'), evt.get('text_fragment', '')[:50], evt.get('event_id'))
                                       for evt in all_events if isinstance(evt, dict)}
                newly_added_count = 0
                for evt_data in entry_events_from_temp:
                    if isinstance(evt_data, dict):
                        key = (evt_data.get('entry_id'), evt_data.get('text_fragment', '')[:50], evt_data.get('event_id'))
                        if key not in existing_event_keys:
                            try:
                                evt_data['entry_id'] = evt_data.get('entry_id', current_entry_id)
                                evt_data['source_date'] = evt_data.get('source_date', current_date)
                                validated_event_dict = RevolutionEvent(**ensure_default_values(evt_data)).model_dump()
                                validated_event_dict = enhanced_field_validation(validated_event_dict) # Применяем доп. валидацию
                                all_events.append(validated_event_dict)
                                all_extracted_events_for_metrics.append(validated_event_dict)
                                existing_event_keys.add(key)
                                newly_added_count += 1
                            except Exception as e_pydantic:
                                logger.error(f"Ошибка Pydantic при загрузке из temp для {current_entry_id}: {e_pydantic}")
                if newly_added_count > 0:
                    logger.info(f"Добавлено {newly_added_count} уникальных событий из temp-файла для {current_entry_id}.")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке/обработке temp-файла {temp_file}: {e}. Обрабатываем запись заново.")
            else:
                with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                    json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                    f_last_proc.write(str(current_entry_id))
                continue

        try:
            extracted_events_data = extract_revolution_events(current_entry_id, current_text, current_date, extractor_model, knowledge_map_for_prompt)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"Экстрактор вернул не-словарь для {current_entry_id}: {event_item_data}. Пропускаем.")
                    continue

                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date
                verified_event_item_data = verify_event(event_item_data, verifier_model, current_text, knowledge_map_for_prompt)

                if not isinstance(verified_event_item_data, dict):
                    logger.warning(f"Верификатор вернул не-словарь для {current_entry_id}: {verified_event_item_data}. Пропускаем.")
                    continue

                verified_event_item_data['entry_id'] = current_entry_id
                verified_event_item_data['source_date'] = current_date

                # Применяем расширенную валидацию после верификации и перед Pydantic
                verified_event_item_data = enhanced_field_validation(verified_event_item_data)

                try:
                    validated_event = RevolutionEvent(**verified_event_item_data)
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка валидации Pydantic для события из {current_entry_id}: {e_pydantic}")
                    logger.debug(f"Данные события (Pydantic ошибка): {verified_event_item_data}")
                    try:
                        deep_fixed_data = ensure_default_values(verified_event_item_data.copy()) # ensure_default_values уже применен в verify_event
                        deep_fixed_data['entry_id'] = current_entry_id
                        deep_fixed_data['source_date'] = current_date
                        validated_event = RevolutionEvent(**deep_fixed_data)
                        processed_entry_events.append(validated_event.model_dump())
                        logger.info(f"Событие для {current_entry_id} добавлено после глубокого исправления Pydantic ошибки.")
                    except Exception as e_pydantic_retry:
                        logger.error(f"Не удалось исправить Pydantic ошибку для {current_entry_id} (финальная попытка): {e_pydantic_retry}")

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(processed_entry_events, f, ensure_ascii=False, indent=2)

            all_events.extend(processed_entry_events)
            all_extracted_events_for_metrics.extend(processed_entry_events)


            with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_events, f, ensure_ascii=False, indent=2)
            with open(LAST_PROCESSED_FILE, "w") as f:
                f.write(str(current_entry_id))

            logger.info(f"Запись {current_entry_id} обработана. Найдено {len(processed_entry_events)} событий.")

        except Exception as e_main_loop:
            logger.error(f"Критическая ошибка при обработке записи {current_entry_id}: {e_main_loop}")
            logger.exception("Traceback критической ошибки в основном цикле:")
            logger.info(f"Пропускаем запись {current_entry_id} из-за ошибки и продолжаем...")
            with open(LAST_PROCESSED_FILE, "w") as f:
                f.write(str(current_entry_id))
            time.sleep(5)

    try:
        with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_events, f, ensure_ascii=False, indent=2)
        logger.info(f"Обработка завершена. Найдено {len(all_events)} событий. Результаты в {FINAL_RESULTS_FILE}")

        # Вывод метрик качества по всем обработанным (или загруженным + обработанным) событиям
        if all_events: # Считаем метрики по всем событиям, которые есть в all_events
            # Для корректного text_coverage нужно суммировать длину всех текстов дневника,
            # которые соответствуют событиям в all_events. Это сложнее, если all_events содержит
            # результаты предыдущих запусков. Проще считать по all_extracted_events_for_metrics,
            # которые были обработаны в этом запуске.
            # Если last_processed_id == 0, то total_text_length_for_metrics будет суммой длин всех текстов.
            # Иначе, это будет сумма длин только новых текстов.
            # Для простоты и демонстрации, посчитаем по all_extracted_events_for_metrics и total_text_length_for_metrics (длина только новых).
            # Если нужен coverage по всему датасету, нужно будет отдельно суммировать длину всех текстов в data.

            logger.info("Расчет метрик качества для событий, обработанных в текущем запуске...")
            metrics = quality_metrics(all_extracted_events_for_metrics, total_text_length_for_metrics)
            logger.info(f"Метрики качества: {json.dumps(metrics, ensure_ascii=False, indent=2, cls=NpEncoder)}")

    except IOError as e:
        logger.error(f"Не удалось сохранить финальный файл {FINAL_RESULTS_FILE}: {e}")

# Класс для сериализации numpy объектов в JSON (для метрик)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Counter):
            return dict(obj)
        return super(NpEncoder, self).default(obj)

# -----------------------------------------------------------------------------
# ТОЧКА ВХОДА
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Запуск обработки дневника...")
    try:
        process_diary()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка на верхнем уровне: {e}")
        logger.exception("Полный traceback неперехваченной ошибки:")
    logger.info("Работа скрипта завершена.")

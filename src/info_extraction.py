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

DATA_PATH = "data/diary_with_id.csv"
KNOWLEDGE_MAP_PATH = "knowledge_map.json" # Путь к JSON-файлу с картой знаний
TEMP_DIR = "temp" # Директория для временных файлов обработки записей
LAST_PROCESSED_FILE = "last_processed.txt" # Файл для отслеживания последней обработанной записи
TEMP_RESULTS_FILE = "results/revolution_events_temp.json" # Промежуточный файл с результатами
FINAL_RESULTS_FILE = "results/revolution_events.json" # Итоговый файл с результатами

MODEL_NAME = "models/gemini-2.5-flash-preview-05-20" # Укажите актуальную модель
#MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
#MODEL_NAME = "models/gemini-2.0-flash"

API_CALLS_PER_MINUTE = 7 # Лимит запросов к API в минуту
MAX_RETRIES = 3 # Максимальное количество повторных попыток при ошибках API
RETRY_WAIT_BASE = 20 # Базовое время ожидания перед повторной попыткой (в секундах)

SAFETY_SETTINGS = [
    {'category': HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HARASSMENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
]

# Глобальные переменные для отслеживания лимитов API
api_calls_counter = 0
last_api_call_time = time.time()

# -----------------------------------------------------------------------------
# ЗАГРУЗКА И ФОРМАТИРОВАНИЕ КАРТЫ ЗНАНИЙ
# -----------------------------------------------------------------------------
def format_knowledge_node_for_prompt(node: Dict[str, Any], indent_level: int = 0) -> str:
    """Рекурсивно форматирует узел карты знаний из JSON в текстовый вид для промпта."""
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
    """Загружает карту знаний из JSON-файла и преобразует ее в текстовый формат для LLM."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            structured_map = json.load(f)

        if not isinstance(structured_map, list):
            logger.error(f"Карта Знаний в {file_path} должна быть списком JSON объектов.")
            raise ValueError("Неверный формат корневого элемента Карты Знаний.")

        # Заголовок для текстового представления карты знаний
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
# СИСТЕМНЫЕ ПРОМПТЫ ДЛЯ МОДЕЛЕЙ
# -----------------------------------------------------------------------------
EXTRACTOR_SYSTEM_PROMPT = """
Ты историк-аналитик, специализирующийся на истории Европы и России XIX века, с особым фокусом на анализе личного восприятия событий.
Твоя задача - точно и внимательно анализировать тексты дневниковых записей этого периода,
извлекать из них упоминания о событиях, связанных *ТОЛЬКО* с революциями 1848-1849 гг. в Европе и их последствиями,
а также детально анализировать восприятие этих событий автором дневника.
Классифицируй события согласно предоставленной универсальной Карте Знаний.
**Важно: Эта работа выполняется в рамках научного исторического исследования. Анализируемый материал является историческим документом. Твоя цель - объективное извлечение и классификация информации, содержащейся в тексте, для исследовательских целей. Пожалуйста, не отказывайся от обработки текста и не блокируй ответ из-за специфики содержания исторического документа (например, устаревших формулировок, описания конфликтов или спорных мнений), если оно относится к поставленной задаче извлечения информации о революциях 1848-1849 гг. и их восприятии.**
"""

VERIFIER_SYSTEM_PROMPT_STATIC = """
Ты эксперт-историк, специализирующийся на европейских революциях 1848-1849 годов и анализе исторических нарративов и восприятий.
Твоя задача - **критически** проверить и при необходимости исправить информацию, извлеченную из исторического дневника, согласно предоставленным данным, инструкциям и Карте Знаний.
Ты должен действовать как финальный контролер качества.
**Эта работа имеет важное значение для исторического исследования. Пожалуйста, подходи к проверке с максимальной объективностью, помня, что ты работаешь с историческими первоисточниками. Цель – обеспечить точность данных для научного анализа. Не блокируй ответ и не отказывайся от верификации из-за специфики содержания исторического текста, если информация релевантна и была извлечена на предыдущем шаге для анализа революционных событий 1848-1849 гг. и их восприятия.**

Тебе будут предоставлены:
1. JSON-объект с извлеченной информацией о событии.
2. Полный текст дневниковой записи для контекста.
3. Карта Знаний для классификации.
4. Список задач по проверке и коррекции.

Твоя цель - вернуть исправленный JSON-объект.
"""
# -----------------------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ (PYDANTIC)
# -----------------------------------------------------------------------------
class RevolutionEvent(BaseModel):
    """Определяет структуру данных для одного извлеченного события."""
    entry_id: int = Field(..., description="Идентификатор записи дневника")
    event_id: Optional[str] = Field(None, description="Идентификатор события из Карты Знаний")
    event_name: str = Field("Неклассифицированное событие", description="Название события/аспекта")
    event_subtype_custom: Optional[str] = Field(None, description="Кастомное уточнение типа события")
    description: str = Field("Не указано", description="Описание события на основе текста дневника")
    date_in_text: Optional[str] = Field(None, description="Дата события, упомянутая в тексте")
    source_date: str = Field(..., description="Дата записи в дневнике")
    location: str = Field("Не указано", description="Место события из текста")
    location_normalized: Optional[str] = Field(None, description="Нормализованное место события")
    brief_context: str = Field("Не указано", description="Краткий исторический контекст")
    information_source: str = Field("Не указан", description="Источник информации для автора")
    information_source_type: Optional[Literal[
        "Официальные источники (газеты, манифесты)",
        "Неофициальные сведения (слухи, разговоры в обществе)",
        "Личные наблюдения и опыт автора",
        "Информация от конкретного лица (именованный источник)",
        "Источник неясен/не указан"
    ]] = Field(None, description="Категория источника информации")
    confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уверенность в извлеченных данных")
    classification_confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уверенность в классификации")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова")
    text_fragment: str = Field("Не указано", description="Цитата из текста дневника")

# -----------------------------------------------------------------------------
# УТИЛИТЫ ДЛЯ РАБОТЫ С API
# -----------------------------------------------------------------------------
def initialize_models():
    """Инициализирует и возвращает модели Gemini для извлечения и верификации."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    extractor_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=EXTRACTOR_SYSTEM_PROMPT
    )
    verifier_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=VERIFIER_SYSTEM_PROMPT_STATIC
    )
    return extractor_model, verifier_model

def manage_api_rate_limit():
    """Управляет частотой запросов к API, чтобы не превышать лимиты."""
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
def ensure_default_values(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Гарантирует наличие значений по умолчанию и корректность типов для полей события."""
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
            "Источник неясен/не указан" # Соответствует промпту экстрактора
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

def load_diary_data(file_path: str) -> pd.DataFrame:
    """Загружает данные дневника из CSV-файла."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ВЕРИФИКАЦИИ ДАННЫХ
# -----------------------------------------------------------------------------
def extract_revolution_events(entry_id: int, text: str, date: str, extractor_model, current_knowledge_map_str: str) -> List[Dict[str, Any]]:
    """Извлекает события из текста дневниковой записи с помощью LLM."""
    manage_api_rate_limit()
    prompt = f"""
    Проанализируй следующую запись из дневника (ID: {entry_id}) от {date}:
    "{text}"

    ---
    **КАРТА ЗНАНИЙ ДЛЯ КЛАССИФИКАЦИИ:**
    {current_knowledge_map_str}
    ---
    **ЗАДАЧА:**
    Твоя задача - внимательно прочитать текст дневниковой записи и извлечь из него ВСЕ упоминания, ПРЯМО связанные с революциями 1848-1849 гг. в Европе, их последствиями, а также реакцией и восприятием этих событий автором дневника. Для КАЖДОГО такого найденного упоминания (события/аспекта) сформируй JSON-объект со следующими полями:

    1.  `entry_id`: Используй предоставленный ID записи: {entry_id}.
    2.  `event_id`:
        *   Определи наиболее подходящий ID из предоставленной Карты Знаний.
        *   Если ни один ID точно не подходит, но упоминание все же связано с революциями 1848-1849 гг., используй 'OTHER_1848'.
        *   Если подходящий ID не найден, но событие релевантно и относится к OTHER_1848, или если ID не может быть однозначно определен, установи `null`.
    3.  `event_name`:
        *   Если `event_id` определен, возьми точное название события/аспекта из Карты Знаний, соответствующее этому `event_id` (без префиксов типа 'Событие:').
        *   Если `event_id` равен 'OTHER_1848' или `null`, сформулируй краткое (2-5 слов) кастомное название, отражающее суть упоминания.
    4.  `event_subtype_custom`:
        *   Если `event_id` или `event_name` слишком общие (например, для категорий _DISCUSS, _GEN, _REFL, _EMO_GENERAL из Карты Знаний, или для `event_id`='OTHER_1848'), предоставь здесь краткое (2-5 слов) уточнение сути события/аспекта, извлеченное непосредственно из текста.
        *   В остальных случаях установи `null`.
    5.  `description`:
        *   Предоставь детальное описание события или аспекта восприятия, СВОИМИ СЛОВАМИ, но строго на основе информации, содержащейся в `text_fragment` и общем контексте записи. Описание должно объяснять, почему это упоминание связано с революциями 1848-1849 гг.
    6.  `date_in_text`:
        *   Если в тексте записи явно указана дата, относящаяся к описываемому событию (не дата самой записи), укажи ее здесь.
        *   В противном случае установи `null`.
    7.  `source_date`: Используй предоставленную дату дневниковой записи: "{date}".
    8.  `location`:
        *   Укажи место события так, как оно упомянуто или подразумевается в тексте.
        *   Если место не указано или неясно, используй "Не указано".
    9.  `location_normalized`:
        *   Укажи нормализованное основное географическое название (город или страна/регион) НА РУССКОМ ЯЗЫКЕ, к которому относится событие. Например, "Франция", "Вена", "Венгрия", "Вологда".
        *   Если определить невозможно, установи `null`.
    10. `brief_context`:
        *   Предоставь очень краткий (1-2 предложения) КОНКРЕТНЫЙ исторический факт, который помогает понять контекст упоминания в дневнике. Это не должно быть пересказом текста дневника или мнением автора.
        *   Если контекст не требуется или неясен, используй "Не указано".
    11. `information_source`:
        *   Опиши источник, из которого автор дневника узнал о событии, как это указано или подразумевается в тексте (например, "газеты", "разговоры с N.", "письмо от брата").
        *   Если источник не указан, используй "Не указан".
    12. `information_source_type`:
        *   Выбери ОДНО из следующих ТОЧНЫХ значений: "{'Официальные источники (газеты, манифесты)', 'Неофициальные сведения (слухи, разговоры в обществе)', 'Личные наблюдения и опыт автора', 'Информация от конкретного лица (именованный источник)', 'Источник неясен/не указан'}".
        *   Если тип источника неясен или не указан, установи `null`.
    13. `confidence`:
        *   Оцени общую уверенность ("High", "Medium", "Low") в корректности всех извлеченных данных для этого события (кроме `event_id` и `classification_confidence`). Учитывай ясность текста, полноту информации.
    14. `classification_confidence`:
        *   Оцени уверенность ("High", "Medium", "Low") в правильности присвоенного `event_id`.
    15. `keywords`:
        *   Извлеки список из от 3 до 5 ключевых слов или коротких фраз, которые наилучшим образом характеризуют событие/аспект.
    16. `text_fragment`:
        *   Извлеки ТОЧНУЮ цитату из текста дневника – ОДНО или НЕСКОЛЬКО ПОЛНЫХ ПРЕДЛОЖЕНИЙ, – которая содержит упоминание о событии и дает достаточный контекст для его понимания. Избегай слишком коротких обрывков.

    **СТРОГИЕ КРИТЕРИИ ОТБОРА (ПОВТОРНО):**
    Извлекай ТОЛЬКО упоминания, которые ПРЯМО связаны с революциями 1848-1849 гг. в Европе.
    ВКЛЮЧАТЬ: Конкретные революционные события, реакции российского правительства, обсуждения в обществе, личные размышления автора о них.
    НЕ ВКЛЮЧАТЬ: Общие рассуждения о политике без привязки к 1848-1849, события вне этого периода, внутренние российские дела без связи с революциями, бытовые упоминания без революционного контекста.
    ПРОВЕРОЧНЫЙ ВОПРОС: "Можно ли это упоминание ПРЯМО связать с революциями 1848-1849 гг.?" Если ответ "нет" или "возможно" - НЕ включай.

    **ФОРМАТ ОТВЕТА:**
    Верни ТОЛЬКО JSON массив, содержащий объекты для каждого найденного релевантного события. Если релевантных упоминаний нет, верни пустой JSON массив `[]`.

    **ПРИМЕР СТРУКТУРЫ ОДНОГО ОБЪЕКТА В МАССИВЕ (используй как шаблон для каждого найденного события):**
    ```json
    {{
        "entry_id": {entry_id},
        "event_id": "REV1848_FRA_FEB",
        "event_name": "Февральская революция / Свержение монархии",
        "event_subtype_custom": null,
        "description": "Автор дневника сообщает о слухах, что во Франции произошли беспорядки, и король бежал, что соответствует событиям Февральской революции.",
        "date_in_text": null,
        "source_date": "{date}",
        "location": "Франция",
        "location_normalized": "Франция",
        "brief_context": "Февральская революция 1848 года во Франции привела к отречению короля Луи-Филиппа I и провозглашению Второй республики.",
        "information_source": "Слухи, разговоры в обществе",
        "information_source_type": "Неофициальные сведения (слухи, разговоры в обществе)",
        "confidence": "High",
        "classification_confidence": "High",
        "keywords": ["Франция", "беспорядки", "король бежал", "революция"],
        "text_fragment": "Слышал сегодня, что во Франции опять беспорядки. Говорят, король бежал."
    }}
    ```
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = extractor_model.generate_content(
                prompt,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.5,
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
    """Верифицирует и корректирует извлеченное событие с помощью LLM."""
    manage_api_rate_limit()
    if not isinstance(event_data, dict):
        logger.error(f"verify_event получил не словарь: {type(event_data)}.")
        return event_data

    event_data_to_verify = ensure_default_values(event_data.copy())
    original_event_id = event_data_to_verify.get('event_id')
    original_text_fragment = event_data_to_verify.get('text_fragment')

    user_prompt_for_verifier = f"""
    Проверь и при необходимости исправь следующую информацию о событии, извлеченную из дневника:
    ```json
    {json.dumps(event_data_to_verify, ensure_ascii=False, indent=2)}
    ```

    Полный текст дневниковой записи для контекста:
    "{full_text}"

    Используй следующую Карту Знаний для проверки классификации (`event_id` и `event_name`):
    {current_knowledge_map_str}

    Выполни следующие задачи по проверке и коррекции JSON-объекта (согласно твоей роли и предоставленной Карте Знаний):
    1.  **Классификация и Уверенность:**
        *   Проверь `event_id`: должен точно соответствовать Карте Знаний и содержанию `text_fragment`. Исправь при необходимости.
        *   Проверь `event_name`: должен соответствовать названию из Карты Знаний для выбранного `event_id` или быть осмысленным кастомным названием для 'OTHER_1848'/null.
        *   Оцени и при необходимости скорректируй `classification_confidence`.

    2.  **Содержание и Текстовая Основа:**
        *   Проверь `description`: должно точно и полно отражать информацию из `text_fragment`.
        *   Проверь `text_fragment`: должен содержать одно или несколько **полных** предложений из оригинального текста, дающих достаточный контекст.
        *   Проверь `keywords`: должны быть релевантными ключевыми словами.

    3.  **Атрибуты События:**
        *   Проверь `event_subtype_custom`: если `event_id` общий, должно быть краткое (2-5 слов) уточнение из текста. Иначе `null`.
        *   Проверь `location` (из текста) и `location_normalized` (нормализованное, русский язык, город/страна, или `null`).
        *   Проверь `date_in_text`: явная дата события из текста, или `null`.

    4.  **Источник Информации:**
        *   Проверь `information_source` (описание из текста).
        *   Проверь `information_source_type`: должно быть ОДНИМ из ТОЧНЫХ значений: "Официальные источники (газеты, манифесты)", "Неофициальные сведения (слухи, разговоры в обществе)", "Личные наблюдения и опыт автора", "Информация от конкретного лица (именованный источник)", "Источник неясен/не указан". Если неясно/неверно, установи `null`.

    5.  **Контекст и Общая Уверенность:**
        *   Проверь `brief_context`: должен быть **конкретным историческим фактом** (1-2 предложения), а не мнением или пересказом. Если нерелевантно, "Не указано".
        *   Оцени и при необходимости скорректируй `confidence` (общая уверенность в данных, кроме классификации).

    **Формат ответа:**
    Верни исправленную версию события ТОЛЬКО в формате JSON объекта. Не добавляй никакого текста до или после JSON.
    """

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = verifier_model.generate_content(
                user_prompt_for_verifier,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.5,
                    response_mime_type="application/json"
                )
            )
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
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДНЕВНИКА
# -----------------------------------------------------------------------------
def process_diary():
    """Основной конвейер обработки дневниковых записей."""
    data = load_diary_data(DATA_PATH)
    if data.empty:
        logger.error("Не удалось загрузить данные дневника или файл пуст. Завершение работы.")
        return

    logger.info("Инициализация моделей LLM...")
    try:
        extractor_model, verifier_model = initialize_models()
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации моделей: {e}. Завершение работы.")
        return

    all_events: List[Dict[str, Any]] = []

    for dir_path in [TEMP_DIR, os.path.dirname(TEMP_RESULTS_FILE), os.path.dirname(FINAL_RESULTS_FILE)]:
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Создана директория {dir_path}")
            except OSError as e:
                logger.error(f"Не удалось создать директорию {dir_path}: {e}. Завершение работы.")
                return

    last_processed_id = 0
    if os.path.exists(LAST_PROCESSED_FILE):
        try:
            with open(LAST_PROCESSED_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    last_processed_id = int(content)
            logger.info(f"Найдена информация о последней обработанной записи: {last_processed_id}")
        except (ValueError, IOError) as e:
            logger.warning(f"Не удалось прочитать ID из {LAST_PROCESSED_FILE}: {e}. Начинаем обработку с начала.")

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
            logger.warning(f"Пропуск строки DataFrame с индексом {index} из-за отсутствующих данных.")
            continue

        current_entry_id = int(current_entry_id_raw)
        current_text = str(current_text)

        if current_entry_id <= last_processed_id:
            continue

        logger.info(f"Обработка записи {current_entry_id} от {current_date}...")
        temp_file_path = os.path.join(TEMP_DIR, f"entry_{current_entry_id}.json")

        if os.path.exists(temp_file_path):
            logger.info(f"Запись {current_entry_id} уже обработана (temp-файл), загружаем...")
            try:
                with open(temp_file_path, "r", encoding="utf-8") as f_temp_entry:
                    entry_events_from_temp = json.load(f_temp_entry)

                existing_event_keys = {(evt.get('entry_id'), evt.get('text_fragment', '')[:50], evt.get('event_id'))
                                       for evt in all_events if isinstance(evt, dict)}
                newly_added_count = 0
                for evt_data in entry_events_from_temp:
                    if isinstance(evt_data, dict):
                        key = (evt_data.get('entry_id', current_entry_id),
                               evt_data.get('text_fragment', '')[:50],
                               evt_data.get('event_id'))
                        if key not in existing_event_keys:
                            try:
                                evt_data['entry_id'] = evt_data.get('entry_id', current_entry_id)
                                evt_data['source_date'] = evt_data.get('source_date', current_date)
                                validated_event = RevolutionEvent(**ensure_default_values(evt_data))
                                all_events.append(validated_event.model_dump())
                                existing_event_keys.add(key)
                                newly_added_count += 1
                            except Exception as e_pydantic:
                                logger.error(f"Ошибка Pydantic при загрузке из temp {current_entry_id}: {e_pydantic}")
                if newly_added_count > 0:
                    logger.info(f"Добавлено {newly_added_count} событий из temp-файла для {current_entry_id}.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки temp-файла {temp_file_path}: {e}. Обрабатываем заново.")
            else:
                try:
                    with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                        json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                    with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                        f_last_proc.write(str(current_entry_id))
                except IOError as e_save:
                    logger.error(f"Ошибка сохр. состояния после загрузки temp для {current_entry_id}: {e_save}")
                continue

        try:
            extracted_events_data = extract_revolution_events(current_entry_id, current_text, current_date, extractor_model, knowledge_map_for_prompt)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"Экстрактор вернул не-словарь для {current_entry_id}: {event_item_data}.")
                    continue

                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date

                verified_event_item_data = verify_event(event_item_data, verifier_model, current_text, knowledge_map_for_prompt)

                if not isinstance(verified_event_item_data, dict):
                    logger.warning(f"Верификатор вернул не-словарь для {current_entry_id}: {verified_event_item_data}.")
                    continue

                verified_event_item_data['entry_id'] = current_entry_id
                verified_event_item_data['source_date'] = current_date

                try:
                    validated_event = RevolutionEvent(**verified_event_item_data)
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка Pydantic для события из {current_entry_id}: {str(e_pydantic)}")
                    logger.debug(f"Данные события (Pydantic ошибка): {verified_event_item_data}")
                    try:
                        deep_fixed_data = ensure_default_values(verified_event_item_data.copy())
                        deep_fixed_data['entry_id'] = current_entry_id
                        deep_fixed_data['source_date'] = current_date
                        validated_event = RevolutionEvent(**deep_fixed_data)
                        processed_entry_events.append(validated_event.model_dump())
                        logger.info(f"Событие для {current_entry_id} добавлено после исправления Pydantic.")
                    except Exception as e_pydantic_final_retry:
                        logger.error(f"Не удалось исправить Pydantic для {current_entry_id}: {str(e_pydantic_final_retry)}")

            try:
                with open(temp_file_path, "w", encoding="utf-8") as f_entry_temp:
                    json.dump(processed_entry_events, f_entry_temp, ensure_ascii=False, indent=2)
            except IOError as e_save_entry:
                 logger.error(f"Ошибка сохранения temp-файла для {current_entry_id}: {e_save_entry}")

            all_events.extend(processed_entry_events)

            try:
                with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                    json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                    f_last_proc.write(str(current_entry_id))
            except IOError as e_save_main:
                 logger.error(f"Ошибка сохр. основного состояния после {current_entry_id}: {e_save_main}")

            logger.info(f"Запись {current_entry_id} обработана. Найдено {len(processed_entry_events)} событий.")

        except Exception as e_main_loop:
            logger.error(f"Критическая ошибка при обработке {current_entry_id}: {str(e_main_loop)}")
            logger.exception(f"Traceback ошибки для {current_entry_id}:")
            logger.info(f"Пропуск записи {current_entry_id} из-за ошибки...")
            try:
                with open(LAST_PROCESSED_FILE, "w") as f:
                    f.write(str(current_entry_id))
            except IOError as e_save_error_state:
                 logger.error(f"Ошибка сохр. LAST_PROCESSED_FILE после ошибки для {current_entry_id}: {e_save_error_state}")
            time.sleep(5)

    try:
        with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f_final:
            json.dump(all_events, f_final, ensure_ascii=False, indent=2)
        logger.info(f"Обработка завершена. Найдено {len(all_events)} событий. Результаты в {FINAL_RESULTS_FILE}")
    except IOError as e:
        logger.error(f"Не удалось сохранить {FINAL_RESULTS_FILE}: {e}")

# -----------------------------------------------------------------------------
# ТОЧКА ВХОДА
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Запуск скрипта обработки дневника...")
    try:
        process_diary()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка на верхнем уровне: {e}")
        logger.exception("Полный traceback неперехваченной ошибки:")
    finally:
        logger.info("Работа скрипта завершена.")

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
        logging.FileHandler("processing_international_v2.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# -----------------------------------------------------------------------------
load_dotenv()

DATA_PATH = "data/diary_with_id.csv" # Убедитесь, что путь корректен
KNOWLEDGE_MAP_PATH = "knowledge_map_international.json" # Используйте адаптированную карту
TEMP_DIR = "temp_international_v2"
LAST_PROCESSED_FILE = "last_processed_international_v2.txt"
TEMP_RESULTS_FILE = "results/international_events_v2_temp.json"
FINAL_RESULTS_FILE = "results/international_events_v2.json"

MODEL_NAME = "models/gemini-2.5-flash-preview-05-20" # Укажите актуальную модель
#MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
#MODEL_NAME = "models/gemini-2.0-flash"


API_CALLS_PER_MINUTE = 10
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
# ЗАГРУЗКА И ФОРМАТИРОВАНИЕ КАРТЫ ЗНАНИЙ
# -----------------------------------------------------------------------------
def format_knowledge_node_for_prompt(node: Dict[str, Any], indent_level: int = 0) -> str:
    indent = "    " * indent_level
    name_prefix = "**" if indent_level == 0 else "*"
    name_suffix = "**" if indent_level == 0 else "*"
    id_str = f" (ID: {node['id']})" if 'id' in node else ""
    line = f"{indent}{name_prefix}   {node['name']}{id_str}{name_suffix}\n"

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
        formatted_string = "**Карта Знаний: Международные События 1848-1849 гг.:**\n\n"
        for top_level_node in structured_map:
            formatted_string += format_knowledge_node_for_prompt(top_level_node, indent_level=0)
        logger.info(f"Карта Знаний (международные события) успешно загружена и отформатирована из {file_path}")
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
Ты историк-аналитик, специализирующийся на европейской истории XIX века, с фокусом на международных отношениях и конфликтах.
Твоя задача - точно и внимательно анализировать тексты дневниковых записей, извлекая из них упоминания *ИСКЛЮЧИТЕЛЬНО о МЕЖДУНАРОДНЫХ СОБЫТИЯХ* периода 1848-1849 гг. (революции в разных странах, войны между государствами, международные интервенции, значимые дипломатические действия).
НЕ ИЗВЛЕКАЙ: внутреннюю политику отдельных стран (если она не имеет прямого международного резонанса), общественные настроения, личное восприятие автора, экономические или культурные события локального значения. Фокусируйся на фактах событий.
Классифицируй события согласно предоставленной Карте Знаний (Международные События 1848-1849 гг.).
**Важно: Анализируемый материал является историческим документом. Твоя цель - объективное извлечение и классификация информации о международных событиях. Пожалуйста, не отказывайся от обработки текста, если он относится к поставленной задаче.**
"""

VERIFIER_SYSTEM_PROMPT_STATIC = """
Ты эксперт-историк, специализирующийся на международных отношениях и конфликтах Европы в середине XIX века, особенно на периоде революций 1848-1849 гг.
Твоя задача - **критически** проверить и при необходимости исправить информацию о МЕЖДУНАРОДНЫХ СОБЫТИЯХ, извлеченную из исторического дневника.
Ты должен действовать как финальный контролер качества, обеспечивая точность и релевантность данных именно международным событиям.
**Важно: Подходи к проверке с максимальной объективностью. Цель – обеспечить точность данных о международных событиях для научного анализа. Не блокируй ответ, если информация релевантна.**

Тебе будут предоставлены:
1. JSON-объект с извлеченной информацией о международном событии.
2. Полный текст дневниковой записи для контекста.
3. Карта Знаний (Международные События 1848-1849 гг.) для классификации.
4. Список задач по проверке и коррекции.

Твоя цель - вернуть исправленный JSON-объект, содержащий ТОЛЬКО информацию о международных событиях.
"""
# -----------------------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ (PYDANTIC) - ОБНОВЛЕНА
# -----------------------------------------------------------------------------
class InternationalEvent(BaseModel):
    """Определяет структуру данных для одного извлеченного международного события."""
    entry_id: int = Field(..., description="Идентификатор записи дневника")
    event_id: Optional[str] = Field(None, description="Идентификатор события из Карты Знаний")
    event_name: str = Field("Неклассифицированное международное событие", description="Название события/аспекта")
    source_date: str = Field(..., description="Дата записи в дневнике")
    date_in_text: Optional[str] = Field(None, description="Дата самого события, если упомянута в тексте")
    event_summary_from_text: str = Field("Не указано", description="Краткое изложение сути события на основе текста дневника")
    location: str = Field("Не указано", description="Место события из текста")
    location_normalized: Optional[str] = Field(None, description="Нормализованное место события (город/страна)")
    brief_context: str = Field("Не указано", description="Конкретный исторический факт (1-2 предложения), связанный с упоминанием.")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова")
    text_fragment: str = Field("Не указано", description="Точная цитата (одно или несколько полных предложений) для контекста.")
    confidence_extraction: Literal["High", "Medium", "Low"] = Field("Medium", description="Уверенность в корректности извлеченных данных для этого события")

# -----------------------------------------------------------------------------
# УТИЛИТЫ ДЛЯ РАБОТЫ С API
# -----------------------------------------------------------------------------
def initialize_models():
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
def ensure_default_values_international(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Гарантирует наличие значений по умолчанию для полей международного события."""
    if not isinstance(event_dict, dict):
        logger.warning(f"ensure_default_values_international получил не словарь: {type(event_dict)}. Возвращаем как есть.")
        return event_dict

    event_dict["event_name"] = event_dict.get("event_name") or "Неклассифицированное международное событие"
    event_dict["event_summary_from_text"] = event_dict.get("event_summary_from_text") or "Не указано"
    event_dict["location"] = event_dict.get("location") or "Не указано"
    event_dict["brief_context"] = event_dict.get("brief_context") or "Не указано"
    event_dict["text_fragment"] = event_dict.get("text_fragment") or "Не указано"
    event_dict["keywords"] = event_dict.get("keywords") or []
    event_dict["confidence_extraction"] = event_dict.get("confidence_extraction") or "Medium"

    optional_fields_to_check_for_empty_string = [
        "event_id", "location_normalized", "date_in_text"
    ]
    for key in optional_fields_to_check_for_empty_string:
        if key in event_dict and event_dict[key] == "":
            event_dict[key] = None

    literal_fields_map = {
        "confidence_extraction": ["High", "Medium", "Low"],
    }
    for field, valid_values in literal_fields_map.items():
        current_value = event_dict.get(field)
        if current_value is not None and current_value not in valid_values:
            logger.warning(f"Недопустимое значение '{current_value}' для поля '{field}'. Устанавливаю в 'Medium'.")
            event_dict[field] = "Medium"
    return event_dict

def load_diary_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ВЕРИФИКАЦИИ ДАННЫХ
# -----------------------------------------------------------------------------
def extract_international_events(entry_id: int, text: str, date: str, extractor_model, current_knowledge_map_str: str) -> List[Dict[str, Any]]:
    """Извлекает МЕЖДУНАРОДНЫЕ события из текста дневниковой записи с помощью LLM."""
    manage_api_rate_limit()
    prompt = f"""
    Проанализируй следующую запись из дневника (ID: {entry_id}) от {date}:
    "{text}"

    ---
    **КАРТА ЗНАНИЙ: МЕЖДУНАРОДНЫЕ СОБЫТИЯ 1848-1849 гг.:**
    {current_knowledge_map_str}
    ---
    **ЗАДАЧА:**
    Твоя задача - внимательно прочитать текст и извлечь из него ВСЕ упоминания, относящиеся *ИСКЛЮЧИТЕЛЬНО к МЕЖДУНАРОДНЫМ СОБЫТИЯМ* периода 1848-1849 гг. К таким событиям относятся: революции в различных странах, рассматриваемые как международные явления или влияющие на другие страны; войны между государствами; международные военные интервенции; значимые дипломатические действия и международные договоры.
    НЕ ИЗВЛЕКАЙ: внутреннюю политику отдельных стран (если она не имеет прямого международного резонанса), общественные настроения внутри одной страны, личное восприятие или эмоции автора, экономические или культурные события локального значения. Фокусируйся на фактах событий на международной арене.

    Для КАЖДОГО найденного МЕЖДУНАРОДНОГО события сформируй JSON-объект со следующими полями:
    1.  `entry_id`: {entry_id} (используй предоставленный ID).
    2.  `event_id`: Наиболее подходящий ID из Карты Знаний. Если не подходит или неоднозначно, используй 'OTHER_INTERNATIONAL_1848' или `null`.
    3.  `event_name`: Название события из Карты Знаний или кастомное для 'OTHER_INTERNATIONAL_1848'/`null`.
    4.  `source_date`: "{date}" (используй предоставленную дату записи).
    5.  `date_in_text`: Если в тексте явно указана дата самого события (отличная от даты записи), укажи ее здесь (например, "март 1848", "15 мая"). Иначе `null`.
    6.  `event_summary_from_text`: Кратко (1-2 предложения) СВОИМИ СЛОВАМИ изложи суть упоминаемого международного события, основываясь ИСКЛЮЧИТЕЛЬНО на информации из текста дневника. Это должно пояснить, о каком именно событии идет речь.
    7.  `location`: Место события из текста. Если неясно, "Не указано".
    8.  `location_normalized`: Нормализованное место (город/страна) НА РУССКОМ. Если неясно, `null`.
    9.  `brief_context`: Краткий (1-2 предложения) КОНКРЕТНЫЙ исторический факт, поясняющий международное значение события (не пересказ дневника).
    10. `keywords`: Список из 3-5 ключевых слов/фраз, характеризующих международное событие.
    11. `text_fragment`: ТОЧНАЯ цитата (ОДНО или НЕСКОЛЬКО ПОЛНЫХ ПРЕДЛОЖЕНИЙ) для контекста международного события.
    12. `confidence_extraction`: Оцени общую уверенность ("High", "Medium", "Low") в корректности извлеченных данных для этого события.

    **ФОРМАТ ОТВЕТА:**
    Верни ТОЛЬКО JSON массив объектов. Если нет релевантных МЕЖДУНАРОДНЫХ событий, верни пустой массив `[]`.

    **ПРИМЕР СТРУКТУРЫ ОБЪЕКТА (шаблон):**
    ```json
    {{
        "entry_id": {entry_id},
        "event_id": "REV1848_HUN_RUS_INTERVENTION",
        "event_name": "Российская интервенция в Венгрию",
        "source_date": "{date}",
        "date_in_text": null,
        "event_summary_from_text": "Автор упоминает, что российские войска вошли в Венгрию для подавления восстания.",
        "location": "Венгрия",
        "location_normalized": "Венгрия",
        "brief_context": "Российская империя вмешалась в венгерскую революцию 1848-1849 гг. по просьбе Австрии для подавления восстания.",
        "keywords": ["Россия", "Венгрия", "интервенция", "войска", "подавление"],
        "text_fragment": "Газеты пишут, что наши войска вошли в Венгрию для усмирения мятежников.",
        "confidence_extraction": "High"
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
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                logger.error(f"Запись {entry_id}: Промпт экстрактора (международные) заблокирован, {error_message_detail}")
                raise ValueError(f"Blocked international prompt for entry_id {entry_id}")

            if not response.candidates:
                logger.error(f"Запись {entry_id}: Нет кандидатов в ответе экстрактора (международные).")
                raise ValueError(f"No candidates in international response for entry_id {entry_id}")

            json_str = response.text
            events = json.loads(json_str)
            if not isinstance(events, list):
                logger.error(f"Экстрактор (международные) вернул не список для {entry_id}. Тип: {type(events)}. Ответ: {events}")
                raise TypeError("LLM returned non-list for international event extraction")
            return events
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON от экстрактора (международные) для {entry_id}: {e}. Строка: {json_str}")
            if retry_count >= MAX_RETRIES -1: raise
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка экстракции (международные) для {entry_id} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE * retry_count)
            else:
                logger.error(f"Превышены попытки экстракции (международные) для {entry_id}. Пропускаем.")
                return []
    return []

def verify_international_event(event_data: Dict[str, Any], verifier_model, full_text: str, current_knowledge_map_str: str) -> Dict[str, Any]:
    """Верифицирует и корректирует извлеченное МЕЖДУНАРОДНОЕ событие с помощью LLM."""
    manage_api_rate_limit()
    if not isinstance(event_data, dict):
        logger.error(f"verify_international_event получил не словарь: {type(event_data)}.")
        return event_data

    event_data_to_verify = ensure_default_values_international(event_data.copy())
    original_event_id = event_data_to_verify.get('event_id')

    user_prompt_for_verifier = f"""
    Проверь и при необходимости исправь следующую информацию о МЕЖДУНАРОДНОМ СОБЫТИИ, извлеченную из дневника:
    ```json
    {json.dumps(event_data_to_verify, ensure_ascii=False, indent=2)}
    ```

    Полный текст дневниковой записи для контекста:
    "{full_text}"

    Используй следующую Карту Знаний (Международные События 1848-1849 гг.) для проверки классификации (`event_id` и `event_name`):
    {current_knowledge_map_str}

    Твоя задача - убедиться, что извлеченное событие действительно является МЕЖДУНАРОДНЫМ (революция, война, интервенция, дипломатия между государствами) и соответствует периоду 1848-1849 гг. Исключи упоминания о чисто внутренних делах одной страны, личных переживаниях автора или локальных новостях без международного значения.

    Выполни следующие задачи по проверке и коррекции JSON-объекта:
    1.  **Релевантность и Классификация:**
        *   Убедись, что событие является МЕЖДУНАРОДНЫМ и относится к 1848-1849 гг. Если нет, верни пустой объект `{{}}`.
        *   Проверь `event_id`: должен точно соответствовать Карте Знаний и содержанию `text_fragment`. Исправь при необходимости.
        *   Проверь `event_name`: должен соответствовать названию из Карты Знаний для выбранного `event_id` или быть осмысленным кастомным названием для 'OTHER_INTERNATIONAL_1848'/null.

    2.  **Содержание и Текстовая Основа:**
        *   Проверь `text_fragment`: должен содержать одно или несколько **полных** предложений из оригинального текста, подтверждающих международный характер события.
        *   Проверь `event_summary_from_text`: должно быть кратким (1-2 предложения) изложением сути международного события СВОИМИ СЛОВАМИ на основе `text_fragment`.
        *   Проверь `keywords`: должны быть релевантными ключевыми словами для международного события.

    3.  **Атрибуты События:**
        *   Проверь `date_in_text`: если дата события явно упомянута в тексте, она должна быть здесь. Иначе `null`.
        *   Проверь `location` (из текста) и `location_normalized` (нормализованное, русский язык, город/страна, или `null`), должны отражать географию международного события.
        *   Проверь `brief_context`: должен быть **конкретным историческим фактом** (1-2 предложения), поясняющим международное значение события.

    4.  **Уверенность:**
        *   Оцени и при необходимости скорректируй `confidence_extraction`.

    **Формат ответа:**
    Верни исправленную версию события ТОЛЬКО в формате JSON объекта. Если событие нерелевантно (не международное или не относится к 1848-1849 гг.), верни пустой объект `{{}}`.
    """

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = verifier_model.generate_content(
                user_prompt_for_verifier,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                logger.error(f"Верификация (международные) для entry_id {event_data.get('entry_id')}: Промпт заблокирован, {error_message_detail}")
                raise ValueError("Blocked international verification prompt")

            if not response.candidates:
                logger.error(f"Верификация (международные) для entry_id {event_data.get('entry_id')}: Нет кандидатов в ответе.")
                raise ValueError("No candidates in international verification response")

            json_str = response.text
            verified_event = json.loads(json_str)

            if not verified_event:
                logger.info(f"Верификатор (международные) счел событие из entry_id {event_data.get('entry_id')} нерелевантным.")
                return {}

            if not isinstance(verified_event, dict):
                logger.error(f"Верификатор (международные) вернул не словарь для entry_id {event_data.get('entry_id')}. Тип: {type(verified_event)}. Ответ: {verified_event}")
                raise TypeError("Verifier returned non-dict for international event")

            if verified_event.get('event_id') != original_event_id:
                logger.info(f"Верификатор (международные) изменил event_id с '{original_event_id}' на '{verified_event.get('event_id')}' для entry_id {event_data.get('entry_id')}")

            verified_event['entry_id'] = event_data_to_verify.get('entry_id')
            verified_event['source_date'] = event_data_to_verify.get('source_date')

            return ensure_default_values_international(verified_event)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON от верификатора (международные) для entry_id {event_data.get('entry_id')}: {e}. Строка: {json_str}")
            if retry_count >= MAX_RETRIES - 1: return event_data_to_verify
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка верификации (международные) для entry_id {event_data.get('entry_id')} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE * retry_count)
            else:
                logger.error(f"Превышены попытки верификации (международные) для entry_id {event_data.get('entry_id')}. Возвращаем исходное событие.")
                return event_data_to_verify
    return event_data_to_verify

# -----------------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДНЕВНИКА
# -----------------------------------------------------------------------------
def process_diary():
    data = load_diary_data(DATA_PATH)
    if data.empty:
        logger.error("Не удалось загрузить данные дневника или файл пуст. Завершение работы.")
        return

    logger.info("Инициализация моделей LLM для международных событий...")
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
            logger.info(f"Найдена информация о последней обработанной записи (международные): {last_processed_id}")
        except (ValueError, IOError) as e:
            logger.warning(f"Не удалось прочитать ID из {LAST_PROCESSED_FILE}: {e}. Начинаем обработку с начала.")

    if os.path.exists(TEMP_RESULTS_FILE):
        try:
            with open(TEMP_RESULTS_FILE, "r", encoding="utf-8") as f:
                all_events = json.load(f)
            logger.info(f"Загружено {len(all_events)} уже обработанных международных событий из {TEMP_RESULTS_FILE}.")
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

        logger.info(f"Обработка записи {current_entry_id} от {current_date} для международных событий...")
        temp_file_path = os.path.join(TEMP_DIR, f"entry_international_{current_entry_id}.json")

        if os.path.exists(temp_file_path):
            logger.info(f"Запись {current_entry_id} (международные) уже обработана, загружаем...")
            try:
                with open(temp_file_path, "r", encoding="utf-8") as f_temp_entry:
                    entry_events_from_temp = json.load(f_temp_entry)

                existing_event_keys = {(evt.get('entry_id'), evt.get('text_fragment', '')[:50], evt.get('event_id'))
                                       for evt in all_events if isinstance(evt, dict)}
                newly_added_count = 0
                for evt_data in entry_events_from_temp:
                    if isinstance(evt_data, dict) and evt_data:
                        key = (evt_data.get('entry_id', current_entry_id),
                               evt_data.get('text_fragment', '')[:50],
                               evt_data.get('event_id'))
                        if key not in existing_event_keys:
                            try:
                                evt_data['entry_id'] = evt_data.get('entry_id', current_entry_id)
                                evt_data['source_date'] = evt_data.get('source_date', current_date)
                                validated_event = InternationalEvent(**ensure_default_values_international(evt_data))
                                all_events.append(validated_event.model_dump())
                                existing_event_keys.add(key)
                                newly_added_count += 1
                            except Exception as e_pydantic:
                                logger.error(f"Ошибка Pydantic (международные) при загрузке из temp {current_entry_id}: {e_pydantic}")
                if newly_added_count > 0:
                    logger.info(f"Добавлено {newly_added_count} межд. событий из temp-файла для {current_entry_id}.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки temp-файла {temp_file_path}: {e}. Обрабатываем заново.")
            else:
                try:
                    with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                        json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                    with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                        f_last_proc.write(str(current_entry_id))
                except IOError as e_save:
                    logger.error(f"Ошибка сохр. состояния после загрузки temp (межд.) для {current_entry_id}: {e_save}")
                continue

        try:
            extracted_events_data = extract_international_events(current_entry_id, current_text, current_date, extractor_model, knowledge_map_for_prompt)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"Экстрактор (международные) вернул не-словарь для {current_entry_id}: {event_item_data}.")
                    continue

                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date

                verified_event_item_data = verify_international_event(event_item_data, verifier_model, current_text, knowledge_map_for_prompt)

                if not isinstance(verified_event_item_data, dict):
                    logger.warning(f"Верификатор (международные) вернул не-словарь для {current_entry_id}: {verified_event_item_data}.")
                    continue

                if not verified_event_item_data:
                    continue

                verified_event_item_data['entry_id'] = current_entry_id
                verified_event_item_data['source_date'] = current_date

                try:
                    validated_event = InternationalEvent(**verified_event_item_data)
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка Pydantic (международные) для события из {current_entry_id}: {str(e_pydantic)}")
                    logger.debug(f"Данные события (Pydantic ошибка, международные): {verified_event_item_data}")
                    try:
                        deep_fixed_data = ensure_default_values_international(verified_event_item_data.copy())
                        deep_fixed_data['entry_id'] = current_entry_id
                        deep_fixed_data['source_date'] = current_date
                        validated_event = InternationalEvent(**deep_fixed_data)
                        processed_entry_events.append(validated_event.model_dump())
                        logger.info(f"Международное событие для {current_entry_id} добавлено после исправления Pydantic.")
                    except Exception as e_pydantic_final_retry:
                        logger.error(f"Не удалось исправить Pydantic (международные) для {current_entry_id}: {str(e_pydantic_final_retry)}")

            try:
                with open(temp_file_path, "w", encoding="utf-8") as f_entry_temp:
                    json.dump(processed_entry_events, f_entry_temp, ensure_ascii=False, indent=2)
            except IOError as e_save_entry:
                 logger.error(f"Ошибка сохранения temp-файла (международные) для {current_entry_id}: {e_save_entry}")

            all_events.extend(processed_entry_events)

            try:
                with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                    json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                    f_last_proc.write(str(current_entry_id))
            except IOError as e_save_main:
                 logger.error(f"Ошибка сохр. основного состояния (международные) после {current_entry_id}: {e_save_main}")

            logger.info(f"Запись {current_entry_id} обработана для межд. событий. Найдено {len(processed_entry_events)}.")

        except Exception as e_main_loop:
            logger.error(f"Критическая ошибка при обработке (международные) {current_entry_id}: {str(e_main_loop)}")
            logger.exception(f"Traceback ошибки (международные) для {current_entry_id}:")
            logger.info(f"Пропуск записи {current_entry_id} (международные) из-за ошибки...")
            try:
                with open(LAST_PROCESSED_FILE, "w") as f:
                    f.write(str(current_entry_id))
            except IOError as e_save_error_state:
                 logger.error(f"Ошибка сохр. LAST_PROCESSED_FILE после ошибки (межд.) для {current_entry_id}: {e_save_error_state}")
            time.sleep(5)

    try:
        with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f_final:
            json.dump(all_events, f_final, ensure_ascii=False, indent=2)
        logger.info(f"Обработка межд. событий завершена. Найдено {len(all_events)}. Результаты в {FINAL_RESULTS_FILE}")
    except IOError as e:
        logger.error(f"Не удалось сохранить {FINAL_RESULTS_FILE}: {e}")

# -----------------------------------------------------------------------------
# ТОЧКА ВХОДА
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Запуск скрипта обработки дневника для МЕЖДУНАРОДНЫХ СОБЫТИЙ...")
    try:
        process_diary()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка на верхнем уровне: {e}")
        logger.exception("Полный traceback неперехваченной ошибки:")
    finally:
        logger.info("Работа скрипта (международные события) завершена.")
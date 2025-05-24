import pandas as pd
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time
import os
import random
import traceback
from dotenv import load_dotenv
import logging # <-- Добавлен импорт logging

# -----------------------------------------------------------------------------
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# -----------------------------------------------------------------------------
# Определяем формат сообщений и уровень логирования
# Логи будут выводиться в консоль и в файл 'processing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log", mode='a', encoding='utf-8'), # 'a' для добавления, 'w' для перезаписи
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# -----------------------------------------------------------------------------

# Загрузка переменных окружения
load_dotenv()

# Пути к файлам
DATA_PATH = "data/diary_with_id.csv"
KNOWLEDGE_MAP_PATH = "knowledge_map.json" # <-- Путь к файлу с картой знаний
TEMP_DIR = "temp"
LAST_PROCESSED_FILE = "last_processed.txt"
TEMP_RESULTS_FILE = "revolution_events_temp.json"
FINAL_RESULTS_FILE = "revolution_events.json"

# Настройки API
MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"

API_CALLS_PER_MINUTE = 7
MAX_RETRIES = 3
RETRY_WAIT_BASE = 20 # секунды

# Настройки безопасности для API
SAFETY_SETTINGS = [
    {'category': HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_HARASSMENT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
    {'category': HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 'threshold': HarmBlockThreshold.BLOCK_NONE},
]

# Глобальные переменные для отслеживания API-запросов
api_calls_counter = 0
last_api_call_time = time.time()

# -----------------------------------------------------------------------------
# ЗАГРУЗКА КАРТЫ ЗНАНИЙ
# -----------------------------------------------------------------------------
def load_knowledge_map(file_path: str) -> str:
    """Загружает Карту Знаний из JSON файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        knowledge_map_str = data.get("knowledge_map_string")
        if not knowledge_map_str or not isinstance(knowledge_map_str, str):
            logger.error(f"Ключ 'knowledge_map_string' не найден или имеет неверный тип в файле {file_path}")
            raise ValueError("Ошибка в формате файла Карты Знаний.")
        logger.info(f"Карта Знаний успешно загружена из {file_path}")
        return knowledge_map_str
    except FileNotFoundError:
        logger.error(f"Файл Карты Знаний не найден: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON в файле Карты Знаний: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при загрузке Карты Знаний: {e}")
        raise

knowledge_map = load_knowledge_map(KNOWLEDGE_MAP_PATH)

# -----------------------------------------------------------------------------
# СИСТЕМНЫЕ ПРОМПТЫ ДЛЯ МОДЕЛЕЙ
# -----------------------------------------------------------------------------
extractor_system_prompt = """
Ты историк-аналитик, специализирующийся на истории Европы и России XIX века, с особым фокусом на анализе личного восприятия событий.
Твоя задача - точно и внимательно анализировать тексты дневниковых записей этого периода,
извлекать из них упоминания о событиях, связанных *ТОЛЬКО* с революциями 1848-1849 гг. в Европе и их последствиями,
а также детально анализировать восприятие этих событий автором дневника.
Классифицируй события согласно предоставленной универсальной Карте Знаний.
**Важно: Эта работа выполняется в рамках научного исторического исследования. Анализируемый материал является историческим документом. Твоя цель - объективное извлечение и классификация информации, содержащейся в тексте, для исследовательских целей. Пожалуйста, не отказывайся от обработки текста и не блокируй ответ из-за специфики содержания исторического документа (например, устаревших формулировок, описания конфликтов или спорных мнений), если оно относится к поставленной задаче извлечения информации о революциях 1848-1849 гг. и их восприятии.**
"""

verifier_system_prompt = """
Ты эксперт-историк, специализирующийся на европейских революциях 1848-1849 годов и анализе исторических нарративов и восприятий.
Твоя задача - **критически** проверить и при необходимости исправить информацию, извлеченную из исторического дневника.
Ты должен действовать как финальный контролер качества.
**Эта работа имеет важное значение для исторического исследования. Пожалуйста, подходи к проверке с максимальной объективностью, помня, что ты работаешь с историческими первоисточниками. Цель – обеспечить точность данных для научного анализа. Не блокируй ответ и не отказывайся от верификации из-за специфики содержания исторического текста, если информация релевантна и была извлечена на предыдущем шаге для анализа революционных событий 1848-1849 гг. и их восприятия.**
Ты должен убедиться, что:
1. Классификация события (`event_id`) точно соответствует его описанию и Карте Знаний.
2. Описание события (`description`) точно и основано на тексте дневника (`text_fragment`).
3. Поля `event_subtype_custom`, `location_normalized`, `information_source_type` заполнены корректно, логично и на основе текста. Для поля `information_source_type` должны использоваться ТОЛЬКО русскоязычные значения из списка: "Официальные источники (газеты, манифесты)", "Неофициальные сведения (слухи, разговоры в обществе)", "Личные наблюдения и опыт автора", "Информация от конкретного лица (именованный источник)", "Источник неясен/не указан".
4. Уровень уверенности в классификации (`classification_confidence`) адекватен.
5. Поле `information_source` правильно указывает, откуда автор дневника узнал о событии.
6. Поле `brief_context` содержит конкретный исторический факт (1-2 предложения), а не мнение или общее рассуждение.
7. Поле `text_fragment` содержит одно или несколько полных предложений, дающих достаточный контекст.
"""
# -----------------------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ (PYDANTIC)
# -----------------------------------------------------------------------------
class RevolutionEvent(BaseModel):
    """
    Модель данных для событий, связанных с революциями 1848-1849 гг. и их восприятием.
    """
    entry_id: int = Field(..., description="Идентификатор записи дневника")
    event_id: Optional[str] = Field(None, description="Идентификатор события из Карты Знаний")
    event_name: str = Field("Неклассифицированное событие", description="Название события/аспекта, взятое из Карты Знаний или сформулированное для OTHER_1848/null.")
    event_subtype_custom: Optional[str] = Field(None, description="Более детальное кастомное название/описание типа события, если стандартный event_name по event_id слишком общий (особенно для категорий _DISCUSS, _GEN, _REFL, _EMO_GENERAL).")
    description: str = Field("Не указано", description="Описание события, основанное исключительно на тексте дневника")
    date_in_text: Optional[str] = Field(None, description="Дата события, указанная в тексте")
    source_date: str = Field(..., description="Дата записи в дневнике")
    location: str = Field("Не указано", description="Место события, как оно указано или подразумевается в тексте")
    location_normalized: Optional[str] = Field(None, description="Нормализованное основное место события (например, 'Вологда', 'Петербург', 'Венгрия', 'Франция') для агрегации. Если определить невозможно, то null.")
    brief_context: str = Field("Не указано", description="Конкретный исторический факт (1-2 предложения), непосредственно связанный с упоминанием, помогающий понять его контекст. Без мнений и общих рассуждений.")
    information_source: str = Field("Не указан", description="Источник информации о событии для автора дневника (текстовое описание)")
    information_source_type: Optional[Literal[
        "Официальные источники (газеты, манифесты)",
        "Неофициальные сведения (слухи, разговоры в обществе)",
        "Личные наблюдения и опыт автора",
        "Информация от конкретного лица (именованный источник)",
        "Источник неясен/не указан"
    ]] = Field(None, description="Категоризированный тип источника информации автора.")
    confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уровень уверенности в корректности извлеченных данных (кроме классификации)")
    classification_confidence: Literal["High", "Medium", "Low"] = Field("Medium", description="Уровень уверенности в правильности присвоенного event_id")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова, извлеченные из текстового фрагмента")
    text_fragment: str = Field("Не указано", description="Точный фрагмент текста (одно или несколько ПОЛНЫХ предложений), дающий КОНТЕКСТ для понимания события.")

# -----------------------------------------------------------------------------
# УТИЛИТЫ ДЛЯ РАБОТЫ С API
# -----------------------------------------------------------------------------
def initialize_models():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    extractor_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=extractor_system_prompt
    )
    verifier_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=verifier_system_prompt
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
def ensure_default_values(event_dict):
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
    event_dict["event_subtype_custom"] = event_dict.get("event_subtype_custom") # Может быть None

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
            logger.warning(f"Недопустимое значение '{current_value}' для поля '{field}'. Ожидались: {valid_values}. Устанавливаю в None.")
            event_dict[field] = None
    return event_dict

def load_diary_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ВЕРИФИКАЦИИ ДАННЫХ
# -----------------------------------------------------------------------------
def extract_revolution_events(entry_id, text, date, extractor_model, current_knowledge_map):
    manage_api_rate_limit()
    prompt = f"""
    Проанализируй следующую запись из дневника Кирилла Березкина (гимназиста из Вологды) от {date}:

    "{text}"

    ---
    **СТРОГИЕ КРИТЕРИИ ОТБОРА:**
    Извлекай ТОЛЬКО упоминания, которые ПРЯМО связаны с революциями 1848-1849 гг. в Европе.

    **ВКЛЮЧАТЬ:**
    - Конкретные революционные события 1848-1849 гг. (февральская революция во Франции, венские события, венгерское восстание, итальянские восстания, германские события)
    - Прямые реакции российского правительства на эти события (военные меры, дипломатические действия, внутренние ограничения)
    - Обсуждения этих событий в российском обществе
    - Личные размышления автора об этих конкретных событиях

    **НЕ ВКЛЮЧАТЬ:**
    - Общие рассуждения о политике без привязки к 1848-1849 гг.
    - События до 1848 или после 1849 года
    - Внутренние российские события, не связанные с европейскими революциями
    - Бытовые упоминания европейских стран без революционного контекста
    - Абстрактные философские размышления о свободе, власти и т.д.
    - Литературные или культурные события без политического контекста 1848-1849 гг.

    **ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА:**
    Перед включением каждого упоминания задай себе вопрос: "Можно ли это упоминание ПРЯМО связать с революционными событиями 1848-1849 гг. в Европе?" Если ответ "нет" или "возможно" - НЕ включай.

    **Используй Карту Знаний для event_id и event_name:**
    {current_knowledge_map}

    **ФОРМАТ ОТВЕТА:**
    Верни ТОЛЬКО JSON массив без дополнительных комментариев, пояснений или текста.


    ```json
    {{
        "entry_id": {entry_id},
        "event_id": "ID_из_Карты_Знаний или 'OTHER_1848' или null",
        "event_name": "Название события/аспекта ИЗ КАРТЫ ЗНАНИЙ (без префиксов типа 'Событие:') или кастомное, если event_id='OTHER_1848' или null",
        "event_subtype_custom": "Уточнение для общих event_id (например, 'Опасения по поводу цензуры') или null",
        "description": "Описание события из текста",
        "date_in_text": "Дата из текста или null",
        "source_date": "{date}",
        "location": "Место события (из текста или 'Не указано')",
        "location_normalized": "Нормализованное место (город/страна) или null",
        "brief_context": "Конкретный исторический факт (1-2 предложения), связанный с упоминанием. Без мнений. Если неясно/не нужно, то 'Не указано'.",
        "information_source": "Источник информации для автора (из текста)",
        "information_source_type": "Тип источника: 'Официальные источники (газеты, манифесты)', 'Неофициальные сведения (слухи, разговоры в обществе)', 'Личные наблюдения и опыт автора', 'Информация от конкретного лица (именованный источник)', 'Источник неясен/не указан', или null.",
        "confidence": "High/Medium/Low",
        "classification_confidence": "High/Medium/Low",
        "keywords": ["список", "ключевых", "слов"],
        "text_fragment": "Точная цитата (ОДНО ИЛИ НЕСКОЛЬКО ПОЛНЫХ ПРЕДЛОЖЕНИЙ для КОНТЕКСТА). Избегай коротких обрывков."
    }}
    ```
    Ожидаемый формат: `[ {{...}}, {{...}} ]`.

    Если нет релевантных упоминаний, верни пустой JSON массив `[]`.
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = extractor_model.generate_content(
                prompt,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                block_reason_message_value = getattr(response.prompt_feedback, 'block_reason_message', None)
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                if block_reason_message_value: error_message_detail += f" ({block_reason_message_value})"
                logger.error(f"Ошибка при обработке записи {entry_id}: Промпт заблокирован, {error_message_detail}")
                current_block_reason_name = block_reason_value.name if hasattr(block_reason_value, 'name') else str(block_reason_value)
                if "OTHER" in current_block_reason_name: logger.warning(f"Запись {entry_id} заблокирована с причиной содержащей 'OTHER'. Попытка {retry_count + 1}/{MAX_RETRIES}.")
                raise ValueError(f"Blocked prompt for entry_id {entry_id}, reason: {current_block_reason_name}")

            if not response.candidates:
                feedback_info = ""
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_value = response.prompt_feedback.block_reason
                    block_reason_message_value = getattr(response.prompt_feedback, 'block_reason_message', None)
                    feedback_info = f", feedback: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                    if block_reason_message_value: feedback_info += f" ({block_reason_message_value})"
                logger.error(f"Ошибка при обработке записи {entry_id}: Нет кандидатов в ответе, возможно, из-за блокировки{feedback_info}.")
                raise ValueError(f"No candidates in response for entry_id {entry_id}, possibly blocked{feedback_info}.")

            json_str = response.text
            try:
                events = json.loads(json_str)
                if not isinstance(events, list):
                    logger.error(f"LLM вернула не список для записи {entry_id}. Тип: {type(events)}. Ответ: {events}")
                    raise TypeError("LLM returned non-list for event extraction")
                return events
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON для записи {entry_id}: {e}. Полученная строка: {json_str}")
                raise e
            except TypeError as e: # Already logged above
                raise e # Re-raise to be caught by the outer loop
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка при обработке записи {entry_id} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                wait_time = RETRY_WAIT_BASE * retry_count
                logger.info(f"Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                logger.error(f"Превышено максимальное количество попыток для записи {entry_id}. Пропускаем...")
                return []
    return []

def verify_event(event_data, verifier_model, full_text, current_knowledge_map):
    manage_api_rate_limit()
    if not isinstance(event_data, dict):
        logger.error(f"verify_event получен не словарь: {type(event_data)}. Возвращаем как есть.")
        return event_data

    event_data_to_verify = ensure_default_values(event_data.copy())
    original_event_id = event_data_to_verify.get('event_id')
    original_text_fragment = event_data_to_verify.get('text_fragment')

    prompt = f"""
    Проверь и при необходимости исправь следующую информацию о событии и его восприятии, извлеченную из дневника:

    ```
    {json.dumps(event_data_to_verify, ensure_ascii=False, indent=2)}
    ```

    **Полный текст дневниковой записи для контекста:**
    "{full_text}"

    Используй следующую Карту Знаний для проверки классификации (`event_id`):
    {current_knowledge_map}

    **Твои задачи:**
    1. **Критически оцени `event_id` и `classification_confidence`.** Исправь, если неверно.
    2. Убедись, что `description` точно отражает `text_fragment`.
    3. **Проверь и скорректируй поля:**
        * `event_subtype_custom`: Если `event_id` общий, это поле должно давать краткое (2-5 слов) уточнение сути. Иначе `null`.
        * `location_normalized`: Нормализованное основное место (город/страна). Если не определить, то `null`.
        * `information_source_type`: Категория источника. Убедись, что используется ОДНО из следующих ТОЧНЫХ русских значений: 'Официальные источники (газеты, манифесты)', 'Неофициальные сведения (слухи, разговоры в обществе)', 'Личные наблюдения и опыт автора', 'Информация от конкретного лица (именованный источник)', 'Источник неясен/не указан'. Если не определить или значение неверное, установи `null`.
        * `brief_context`: Убедись, что здесь указан **конкретный исторический факт (1-2 предложения)**, а не мнение или общее рассуждение. Если нет, исправь или установи 'Не указано'.
    4. Убедись, что `event_name` соответствует названию из Карты Знаний для данного `event_id` (без префиксов типа 'Событие:'), или является кастомным для 'OTHER_1848'/null.
    5. Если поле `location` или `information_source` пустое, но информацию можно извлечь из текста, заполни его. Иначе оставь "Не указано" / "Не указан" или `null` для опциональных.
    6. **Проверь `text_fragment`**: Убедись, что это одно или несколько ПОЛНЫХ предложений, дающих достаточный КОНТЕКСТ. При необходимости, можешь заменить или расширить фрагмент, используя полный текст дневниковой записи.

    **Формат ответа:**
    Верни исправленную версию события ТОЛЬКО в формате JSON объекта. Не добавляй никакого текста до или после JSON.
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = verifier_model.generate_content(
                prompt,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GenerationConfig(
                    temperature=0.5,
                    response_mime_type="application/json"
                )
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                block_reason_message_value = getattr(response.prompt_feedback, 'block_reason_message', None)
                error_message_detail = f"причина: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                if block_reason_message_value: error_message_detail += f" ({block_reason_message_value})"
                logger.error(f"Ошибка при верификации события: Промпт заблокирован, {error_message_detail}")
                current_block_reason_name = block_reason_value.name if hasattr(block_reason_value, 'name') else str(block_reason_value)
                if "OTHER" in current_block_reason_name: logger.warning(f"Верификация заблокирована с причиной содержащей 'OTHER'. Попытка {retry_count + 1}/{MAX_RETRIES}.")
                raise ValueError(f"Blocked prompt for verification, reason: {current_block_reason_name}")

            if not response.candidates:
                feedback_info = ""
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_value = response.prompt_feedback.block_reason
                    block_reason_message_value = getattr(response.prompt_feedback, 'block_reason_message', None)
                    feedback_info = f", feedback: {block_reason_value.name if hasattr(block_reason_value, 'name') else block_reason_value}"
                    if block_reason_message_value: feedback_info += f" ({block_reason_message_value})"
                logger.error(f"Ошибка при верификации события: Нет кандидатов в ответе, возможно, из-за блокировки{feedback_info}.")
                raise ValueError(f"No candidates in response for verification, possibly blocked{feedback_info}.")

            json_str = response.text
            try:
                verified_event = json.loads(json_str)
                if not isinstance(verified_event, dict):
                    logger.error(f"Верификатор вернул не словарь. Тип: {type(verified_event)}. Ответ: {verified_event}")
                    raise TypeError("Verifier returned non-dict")
                if verified_event.get('event_id') != original_event_id:
                    logger.info(f"Верификатор изменил event_id с '{original_event_id}' на '{verified_event.get('event_id')}' для entry_id {event_data.get('entry_id')}")
                if verified_event.get('text_fragment') != original_text_fragment:
                    logger.info(f"Верификатор изменил text_fragment для entry_id {event_data.get('entry_id')}")
                return ensure_default_values(verified_event)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON при верификации: {e}. Полученная строка: {json_str}")
                raise e
            except TypeError as e: # Already logged above
                raise e # Re-raise
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка при верификации события (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                wait_time = RETRY_WAIT_BASE * retry_count
                logger.info(f"Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                logger.error("Превышено максимальное количество попыток верификации. Возвращаем исходное событие (после ensure_default_values)...")
                return event_data_to_verify
    return event_data_to_verify

# -----------------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДНЕВНИКА
# -----------------------------------------------------------------------------
def process_diary():
    # Загрузка Карты Знаний (уже загружена глобально, но можно передавать как параметр)
    current_knowledge_map = knowledge_map

    data = load_diary_data(DATA_PATH)
    if data.empty:
        logger.error("Не удалось загрузить данные дневника. Завершение работы.")
        return []

    logger.info("Инициализация моделей...")
    try:
        extractor_model, verifier_model = initialize_models()
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации моделей: {e}. Завершение работы.")
        return []

    all_events = []
    if not os.path.exists(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
            logger.info(f"Создана директория {TEMP_DIR}")
        except OSError as e:
            logger.error(f"Не удалось создать директорию {TEMP_DIR}: {e}. Завершение работы.")
            return []


    last_processed_id = 0
    if os.path.exists(LAST_PROCESSED_FILE):
        try:
            with open(LAST_PROCESSED_FILE, "r") as f:
                last_processed_id = int(f.read().strip())
            logger.info(f"Найдена информация о последней обработанной записи: {last_processed_id}")
        except ValueError:
            logger.warning(f"Не удалось прочитать ID из {LAST_PROCESSED_FILE}. Начинаем с начала.")
        except Exception as e:
            logger.warning(f"Не удалось прочитать информацию о последней обработанной записи из {LAST_PROCESSED_FILE}: {e}. Начинаем с начала.")


    if os.path.exists(TEMP_RESULTS_FILE):
        try:
            with open(TEMP_RESULTS_FILE, "r", encoding="utf-8") as f:
                all_events = json.load(f)
            logger.info(f"Загружено {len(all_events)} уже обработанных событий из {TEMP_RESULTS_FILE}.")
        except Exception as e:
            logger.warning(f"Не удалось загрузить уже обработанные события из {TEMP_RESULTS_FILE}: {e}. Начинаем с нуля.")
            all_events = []

    for index, row in data.iterrows():
        current_entry_id = row['entry_id']
        current_date = row['date']
        current_text = str(row['text']) # Убедимся, что текст - строка

        if pd.isna(current_entry_id) or pd.isna(current_date) or pd.isna(current_text):
            logger.warning(f"Пропуск строки {index} из-за отсутствующих данных (entry_id, date или text).")
            continue

        current_entry_id = int(current_entry_id) # Убедимся, что ID - целое число

        if current_entry_id <= last_processed_id:
            continue

        logger.info(f"Обработка записи {current_entry_id} от {current_date}...")
        temp_file = os.path.join(TEMP_DIR, f"entry_{current_entry_id}.json")

        if os.path.exists(temp_file):
            logger.info(f"Запись {current_entry_id} уже обработана (найден temp-файл), загружаем результаты...")
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    entry_events_from_temp = json.load(f)

                existing_event_keys = set()
                for evt_dict in all_events:
                    if isinstance(evt_dict, dict):
                         key = (evt_dict.get('entry_id'), evt_dict.get('text_fragment', '')[:50], evt_dict.get('event_id'))
                         existing_event_keys.add(key)

                newly_added_count = 0
                for evt_data in entry_events_from_temp:
                    if isinstance(evt_data, dict):
                        key = (evt_data.get('entry_id'), evt_data.get('text_fragment', '')[:50], evt_data.get('event_id'))
                        if key not in existing_event_keys:
                            try:
                                evt_data['entry_id'] = evt_data.get('entry_id', current_entry_id)
                                evt_data['source_date'] = evt_data.get('source_date', current_date)
                                validated_event_from_temp = RevolutionEvent(**ensure_default_values(evt_data))
                                all_events.append(validated_event_from_temp.model_dump())
                                existing_event_keys.add(key)
                                newly_added_count +=1
                            except Exception as e_pydantic_temp:
                                logger.error(f"Ошибка Pydantic при загрузке из temp для {current_entry_id}, событие: {evt_data}, ошибка: {e_pydantic_temp}")
                if newly_added_count > 0:
                    logger.info(f"Добавлено {newly_added_count} уникальных событий из temp-файла для записи {current_entry_id}.")

                with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                    json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                    f_last_proc.write(str(current_entry_id))
                continue
            except Exception as e:
                logger.warning(f"Ошибка при загрузке/обработке temp-файла {temp_file}: {e}. Обрабатываем запись {current_entry_id} заново.")

        try:
            extracted_events_data = extract_revolution_events(current_entry_id, current_text, current_date, extractor_model, current_knowledge_map)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"extract_revolution_events вернул не-словарь для записи {current_entry_id}: {event_item_data}. Пропускаем.")
                    continue

                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date
                verified_event_item_data = verify_event(event_item_data, verifier_model, current_text, current_knowledge_map)

                if not isinstance(verified_event_item_data, dict):
                    logger.warning(f"verify_event вернул не-словарь для записи {current_entry_id}: {verified_event_item_data}. Пропускаем.")
                    continue

                verified_event_item_data['entry_id'] = current_entry_id
                verified_event_item_data['source_date'] = current_date

                try:
                    validated_event = RevolutionEvent(**verified_event_item_data)
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка валидации Pydantic для события из записи {current_entry_id}: {str(e_pydantic)}")
                    logger.debug(f"Данные события перед Pydantic: {verified_event_item_data}")
                    try:
                        if isinstance(verified_event_item_data, dict):
                            deep_fixed_data = ensure_default_values(verified_event_item_data.copy())
                            deep_fixed_data['entry_id'] = current_entry_id
                            deep_fixed_data['source_date'] = current_date
                            validated_event = RevolutionEvent(**deep_fixed_data)
                            processed_entry_events.append(validated_event.model_dump())
                            logger.info(f"Событие для записи {current_entry_id} (попытка глубокого исправления) и добавлено после ошибки Pydantic.")
                        else:
                             logger.warning(f"Не удалось глубоко исправить (не словарь): {verified_event_item_data}")
                    except Exception as e_pydantic_final_retry:
                        logger.error(f"Не удалось исправить событие для записи {current_entry_id} после ошибки Pydantic (финальная попытка): {str(e_pydantic_final_retry)}")

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(processed_entry_events, f, ensure_ascii=False, indent=2)

            all_events.extend(processed_entry_events)

            with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_events, f, ensure_ascii=False, indent=2)
            with open(LAST_PROCESSED_FILE, "w") as f:
                f.write(str(current_entry_id))

            logger.info(f"Запись {current_entry_id} обработана. Найдено {len(processed_entry_events)} событий.")

        except Exception as e_main_loop:
            logger.error(f"Критическая ошибка при обработке записи {current_entry_id}: {str(e_main_loop)}")
            logger.exception("Traceback критической ошибки:") # Логирует полный traceback
            logger.info(f"Пропускаем запись {current_entry_id} из-за критической ошибки и продолжаем...")
            with open(LAST_PROCESSED_FILE, "w") as f: # Записываем, чтобы пропустить эту запись при следующем запуске
                f.write(str(current_entry_id))
            time.sleep(5) # Небольшая пауза на всякий случай

    try:
        with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_events, f, ensure_ascii=False, indent=2)
        logger.info(f"Обработка завершена. Найдено {len(all_events)} событий. Результаты сохранены в {FINAL_RESULTS_FILE}")
    except IOError as e:
        logger.error(f"Не удалось сохранить финальный файл результатов {FINAL_RESULTS_FILE}: {e}")

    return all_events
# -----------------------------------------------------------------------------
# ТОЧКА ВХОДА
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Запуск обработки дневника...")
    try:
        process_diary()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка на верхнем уровне выполнения: {e}")
        logger.exception("Полный traceback неперехваченной ошибки:")
    logger.info("Работа скрипта завершена.")
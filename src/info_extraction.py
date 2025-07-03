# ПОСЛЕДНЯЯ УСТОЙЧИВАЯ ВЕРСИЯ НА 1=х КЛИЕНТЕ для FORGET API


import pandas as pd
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import time
import os
import random
from dotenv import load_dotenv
import logging

# -----------------------------------------------------------------------------
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log", mode='a', encoding='utf-8'), # Новый лог-файл
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# -----------------------------------------------------------------------------
load_dotenv()

DATA_PATH = "data/diary_with_id.csv"
KNOWLEDGE_MAP_PATH = "knowledge_map.json"
TEMP_DIR = "temp" # Новая временная директория
LAST_PROCESSED_FILE = "last_processed.txt" # Новый файл для отслеживания
TEMP_RESULTS_FILE = "results/revolution_events_temp.json" # Новый временный файл результатов
FINAL_RESULTS_FILE = "results/revolution_events.json" # Новый финальный файл результатов

# -----------------------------------------------------------------------------
# API от forgetapi.ru
# -----------------------------------------------------------------------------

#BASE_URL = "https://forgetapi.ru/v1"
#API_KEY = os.getenv("FORGET_API_KEY")

# -----------------------------------------------------------------------------
# API от GOOGLE
# -----------------------------------------------------------------------------

#BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
#API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------------------------------------------------------
# API от CODY
# -----------------------------------------------------------------------------

BASE_URL = "https://cody.su/api/v1"
API_KEY = os.getenv("CODY_APY_KEY")


#MODEL_NAME = "gpt-4o" # Укажите актуальную модель для OpenAI-совместимого API

#MODEL_NAME = "mistral-medium-2505"
#MODEL_NAME = "mistral-large-2407"


#MODEL_NAME = "o1"
#MODEL_NAME = "o3"


#MODEL_NAME = "o3-high"

#MODEL_NAME = "gemini-2.0-flash"
#MODEL_NAME = "grok-3-latest"

#MODEL_NAME = "claude-sonnet-4-20250514-thinking"
#MODEL_NAME = "claude-opus-4-20250514"

#MODEL_NAME = "o4-mini-high"
MODEL_NAME = "gpt-4.1"

#MODEL_NAME = "gemini-2.5-pro"

#MODEL_NAME = "Qwen/Qwen3-235B-A22B"

#MODEL_NAME = "models/gemini-2.5-flash" # Укажите актуальную модель
#MODEL_NAME = "models/gemini-2.0-flash"
#MODEL_NAME = "models/gemini-1.5-flash"




TEMPERATURE = 0.1 # Снижаем температуру для большей точности и следования примерам

API_CALLS_PER_MINUTE = 15
MAX_RETRIES = 3
RETRY_WAIT_BASE = 20

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
        logger.error(f"Непредвиденная ошибка при загрузке Карты Знаний: {e}")
        logger.exception("Traceback ошибки форматирования карты:")
        raise

knowledge_map_for_prompt = load_and_format_knowledge_map(KNOWLEDGE_MAP_PATH)

# -----------------------------------------------------------------------------
# СИСТЕМНЫЙ ПРОМПТ ДЛЯ ЭКСТРАКТОРА (УТОЧНЕННЫЙ)
# -----------------------------------------------------------------------------
EXTRACTOR_SYSTEM_PROMPT = """
Ты высококвалифицированный историк-аналитик, специализирующийся на истории Европы и России XIX века, с особым фокусом на анализе личного восприятия событий.
Твоя задача - предельно точно и внимательно анализировать тексты дневниковых записей этого периода.
Твой анализ проводится в рамках исследования "Эхо 'Весны народов' в российской провинции: восприятие событий 1848-1849 гг. в дневнике Кирилла Березкина".
Извлекай из них упоминания о событиях, связанных *ТОЛЬКО* с революциями 1848-1849 гг. в Европе и их последствиями, а также детально анализируй восприятие этих событий автором дневника.
Классифицируй события согласно предоставленной универсальной Карте Знаний.
Учитывай исторический контекст:
- Автор дневника: Кирилл Антонович Березкин, гимназист Вологодской губернской гимназии, потомственный дворянин.
- Период ведения дневника: январь 1849 - сентябрь 1849.
- Место: город Вологда, Вологодская губерния.
- Историческая значимость: дневник отражает восприятие важных исторических событий в российской провинции.
**Важно: Эта работа выполняется в рамках научного исторического исследования. Анализируемый материал является историческим документом. Твоя цель - объективное извлечение и классификация информации, содержащейся в тексте, для исследовательских целей. Пожалуйста, не отказывайся от обработки текста и не блокируй ответ из-за специфики содержания исторического документа (например, устаревших формулировок, описания конфликтов или спорных мнений), если оно относится к поставленной задаче извлечения информации о революциях 1848-1849 гг. и их восприятии.**

**КЛЮЧЕВЫЕ ПРАВИЛА КЛАССИФИКАЦИИ ВОСПРИЯТИЯ (СТРОГО СЛЕДОВАТЬ!):**
Для максимально точной классификации личного восприятия автора и общественных настроений, пожалуйста, строго следуй этим правилам:
*   **Личное восприятие автора (`AUTHOR_PERCEPTION_...`):** Используй категории, начинающиеся с `AUTHOR_PERCEPTION_`, **ТОЛЬКО** когда автор **ПРЯМО ВЫРАЖАЕТ СВОИ СОБСТВЕННЫЕ** чувства, мнения, суждения, предубеждения, симпатии или антипатии. Это его личная позиция, его внутренний мир. Например, если он *сам* говорит: *"Мне кажется, что поляки..."* или *"Я ужасно зол на..."* или *"Я не одобряю..."*.
*   **Общественные настроения (`RU_REACTION_SOCIAL_...`):** Используй категории, начинающиеся с `RU_REACTION_SOCIAL_`, **ТОЛЬКО** когда автор **СООБЩАЕТ О НАСТРОЕНИЯХ, СЛУХАХ, ДИСКУССИЯХ или МНЕНИЯХ**, которые он слышал или наблюдал в обществе, но **НЕ ОБЯЗАТЕЛЬНО присоединяется к ним явно** или выражает свои эмоции по этому поводу. Это внешняя информация, которую он передает. Например, если он *сообщает*: *"Говорят, что поляки хотят..."*, *"В Петербурге вся молодежь заражена..."*, *"Слухи носятся, что..."*

**ПРАВИЛО ДЛЯ `brief_context`:**
Поле `brief_context` должно содержать **ОДИН-ДВА КРАТКИХ, РЕЛЕВАНТНЫХ ВНЕШНИХ ИСТОРИЧЕСКИХ ФАКТА**, которые помогают понять извлеченное событие. **НЕ ПЕРЕСКАЗЫВАЙ дневник или `description`**. Если релевантный внешний факт трудно подобрать или он не нужен, укажи "Не указано".

Убедись, что твой ответ ВСЕГДА является валидным JSON массивом, даже если он пустой (`[]`). Не добавляй никакого текста до или после JSON. Тщательно проверяй соответствие всех полей запрашиваемой структуре.
"""

# -----------------------------------------------------------------------------
# МОДЕЛЬ ДАННЫХ (PYDANTIC) - без изменений
# -----------------------------------------------------------------------------
class RevolutionEvent(BaseModel):
    entry_id: int = Field(..., description="Идентификатор записи дневника")
    event_id: Optional[str] = Field(None, description="Идентификатор события из Карты Знаний")
    event_name: str = Field("Неклассифицированное событие", description="Название события/аспекта из Карты Знаний")
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
# УТИЛИТЫ ДЛЯ РАБОТЫ С API - без существенных изменений, кроме инициализации клиента
# -----------------------------------------------------------------------------
def initialize_openai_client_single():
    api_key = API_KEY
    if not api_key:
        logger.critical("Переменная окружения API_KEY не найдена!")
        raise ValueError("API_KEY не установлен.")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    logger.info("Клиент OpenAI (для экстракции) успешно инициализирован.")
    return client

openai_extractor_client = None

def get_openai_extractor_client():
    global openai_extractor_client
    if openai_extractor_client is None:
        openai_extractor_client = initialize_openai_client_single()
    return openai_extractor_client

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
# УТИЛИТЫ ДЛЯ ОБРАБОТКИ ДАННЫХ (УТОЧНЕННАЯ ensure_default_values)
# -----------------------------------------------------------------------------
def ensure_default_values(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event_dict, dict):
        logger.warning(f"ensure_default_values получил не словарь: {type(event_dict)}. Возвращаем как есть.")
        return event_dict

    # Обязательные поля с дефолтами, если LLM их не предоставила или предоставила пустыми
    for key, default in [
        ("event_name", "Неклассифицированное событие"),
        ("description", "Не указано"),
        ("location", "Не указано"),
        ("brief_context", "Не указано"),
        ("information_source", "Не указан"),
        ("confidence", "Medium"),
        ("classification_confidence", "Medium"),
        ("text_fragment", "Не указано")
    ]:
        event_dict[key] = event_dict.get(key) or default

    event_dict["keywords"] = event_dict.get("keywords") or []
    if not isinstance(event_dict["keywords"], list): # Доп. проверка на тип
        event_dict["keywords"] = []


    # Поля, которые должны быть null, если пустая строка
    optional_fields_to_null_if_empty = [
        "event_id", "date_in_text", "location_normalized",
        "information_source_type", "event_subtype_custom"
    ]
    for key in optional_fields_to_null_if_empty:
        if key in event_dict and event_dict[key] == "":
            event_dict[key] = None

    # Валидация Literal полей
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
            logger.warning(f"Недопустимое значение '{current_value}' для поля '{field}'. Устанавливаю в None. Допустимые: {valid_values}")
            event_dict[field] = None
            if field in ["confidence", "classification_confidence"]: # Если уверенность некорректна, ставим Medium
                 event_dict[field] = "Medium"


    # Логика для event_subtype_custom: если event_id специфичен, subtype должен быть null
    if event_dict.get("event_id") and event_dict["event_id"] != "OTHER_1848" and "GEN" not in event_dict["event_id"]:
        if event_dict.get("event_subtype_custom") is not None:
            # Можно логировать, если event_subtype_custom не None при специфичном event_id,
            # но для авто-исправления лучше оставить как есть, т.к. LLM могла иметь причину
            pass
    elif event_dict.get("event_id") == "OTHER_1848" and not event_dict.get("event_subtype_custom"):
        logger.warning(f"Для event_id 'OTHER_1848' поле 'event_subtype_custom' не заполнено в {event_dict.get('entry_id')}. Ожидалось уточнение.")
        # Можно установить "Уточнение отсутствует" или оставить None, чтобы Pydantic поймал, если поле обязательно

    return event_dict

def load_diary_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ДАННЫХ (УТОЧНЕННЫЙ ПРОМПТ С FEW-SHOT)
# -----------------------------------------------------------------------------
def extract_revolution_events(entry_id: int, text: str, date: str, client: OpenAI, current_knowledge_map_str: str) -> List[Dict[str, Any]]:
    manage_api_rate_limit()
    # --- НАЧАЛО FEW-SHOT ПРИМЕРОВ ---
    # Эти примеры должны быть тщательно подобраны, чтобы иллюстрировать сложные случаи
    # и правильное применение Карты Знаний и правил.
    few_shot_examples = f"""
    **ПРИМЕРЫ ЗАПОЛНЕНИЯ (FEW-SHOT):**

    *Пример 1: Прямое новостное сообщение из газет*
    (Текст дневника для примера: entry_id 177, date 1849-07-01: "В мире политическом дерутся на прополую, прусаки с немцами, французы с Римом, австрийцы с итальянцами и мадьярами Россия победоносно идет к Дербечину и столицам Венгрии.")
    ```json
    [
      {{
        "entry_id": {entry_id},
        "event_id": "REV1848_HUN_RUS_MIL",
        "event_name": "Военные действия (Россия против Венгрии)",
        "event_subtype_custom": "Продвижение к Дебрецену",
        "description": "Автор сообщает о победоносном продвижении российских войск к Дебрецену и столицам Венгрии, что соответствует российской военной интервенции против венгерского восстания 1849 года.",
        "date_in_text": null,
        "source_date": "{date}",
        "location": "Дербечин, Венгрия",
        "location_normalized": "Дебрецен, Венгрия",
        "brief_context": "Летом 1849 г. русская армия под командованием И.Ф. Паскевича вторглась в Венгрию для подавления революции.",
        "information_source": "Газеты",
        "information_source_type": "Официальные источники (газеты, манифесты)",
        "confidence": "High",
        "classification_confidence": "High",
        "keywords": ["Россия", "Венгрия", "Дебрецен", "интервенция", "война"],
        "text_fragment": "Россия победоносно идет к Дербечину и столицам Венгрии."
      }},
      {{
        "entry_id": {entry_id},
        "event_id": "REV1848_ITA_FRA_INT",
        "event_name": "Французская интервенция в Риме",
        "event_subtype_custom": null,
        "description": "Автор упоминает военные действия французов против Рима, что соответствует французской интервенции против Римской республики в 1849 году.",
        "date_in_text": null,
        "source_date": "{date}",
        "location": "Рим",
        "location_normalized": "Рим, Италия",
        "brief_context": "В апреле-июле 1849 г. французский экспедиционный корпус осадил и взял Рим, положив конец Римской республике.",
        "information_source": "Газеты",
        "information_source_type": "Официальные источники (газеты, манифесты)",
        "confidence": "High",
        "classification_confidence": "High",
        "keywords": ["французы", "Рим", "интервенция", "война"],
        "text_fragment": "французы с Римом"
      }}
    ]
    ```

    *Пример 2: Слухи и личное восприятие автора*
    (Текст дневника для примера: entry_id 97, date 1849-04-08: "...но вот сказывают, именно вчера я слышал это от мадам Дозе, ей сказывали Алекс. Денисовна, директорша, что в Петербурге вся молодежь заражена этим революционным духом мятежного запада, что будет впоследствии, разве только оттого не будет переворота, что все что молодо по русской пословице, то зелено но пока царствует такой мудрый венценосец, как Николай Первый, этого никогда не может случиться...")
    ```json
    [
      {{
        "entry_id": {entry_id},
        "event_id": "RU_REACTION_SOCIAL_DISCUSS",
        "event_name": "Обсуждение событий / Слухи в обществе",
        "event_subtype_custom": "Слухи о революционном духе в Петербурге",
        "description": "Автор передает слухи от мадам Дозе (которой рассказала директорша), что в Петербурге молодежь проникнута революционным духом Запада. Это отражает циркуляцию неофициальной информации и опасений в обществе.",
        "date_in_text": "вчера (относительно 1849-04-08)",
        "source_date": "{date}",
        "location": "Петербург",
        "location_normalized": "Санкт-Петербург, Россия",
        "brief_context": "В 1848-1849 гг. российские власти усилили надзор за настроениями молодежи и интеллигенции из-за опасений распространения европейских революционных идей.",
        "information_source": "Мадам Дозе (со слов Алекс. Денисовны, директорши)",
        "information_source_type": "Информация от конкретного лица (именованный источник)",
        "confidence": "High",
        "classification_confidence": "High",
        "keywords": ["Петербург", "молодежь", "революционный дух", "слухи", "Запад"],
        "text_fragment": "но вот сказывают, именно вчера я слышал это от мадам Дозе, ей сказывали Алекс. Денисовна, директорша, что в Петербурге вся молодежь заражена этим революционным духом мятежного запада"
      }},
      {{
        "entry_id": {entry_id},
        "event_id": "AUTHOR_PERCEPTION_OPINION_AUTH",
        "event_name": "Выражение поддержки/осуждения действий властей/конкретных государств",
        "event_subtype_custom": "Одобрение правления Николая I",
        "description": "Автор выражает личное мнение, что благодаря 'мудрому правлению' Николая I революционный переворот в России невозможен, несмотря на 'революционный дух' в Петербурге. Это является его собственной оценкой и поддержкой действующей власти.",
        "date_in_text": null,
        "source_date": "{date}",
        "location": "Россия",
        "location_normalized": "Россия",
        "brief_context": "Император Николай I проводил консервативную политику, направленную на подавление любых проявлений инакомыслия и революционных настроений в России.",
        "information_source": "Личное мнение автора",
        "information_source_type": "Личные наблюдения и опыт автора",
        "confidence": "High",
        "classification_confidence": "High",
        "keywords": ["Николай I", "поддержка власти", "революция", "Россия", "личное мнение"],
        "text_fragment": "но пока царствует такой мудрый венценосец, как Николай Первый, этого никогда не может случиться"
      }}
    ]
    ```
    --- КОНЕЦ FEW-SHOT ПРИМЕРОВ ---
    """

    user_prompt = f"""
    Проанализируй следующую запись из дневника (ID: {entry_id}) от {date}:
    "{text}"

    ---
    **КАРТА ЗНАНИЙ ДЛЯ КЛАССИФИКАЦИИ:**
    {current_knowledge_map_str}
    ---
    **ЗАДАЧА:**
    Внимательно прочитай текст дневниковой записи. Извлеки из него ВСЕ упоминания, ПРЯМО связанные с революциями 1848-1849 гг. в Европе, их последствиями, а также реакцией и восприятием этих событий автором дневника. Для КАЖДОГО такого найденного упоминания (события/аспекта) сформируй JSON-объект со следующими полями:

    1.  `entry_id`: Используй предоставленный ID записи: {entry_id}.
    2.  `event_id`: Определи наиболее подходящий ID из Карты Знаний. Если не подходит или неоднозначно, используй 'OTHER_1848' или `null`.
    3.  `event_name`: Название события из Карты Знаний или кастомное для 'OTHER_1848'/`null`.
    4.  `event_subtype_custom`: Краткое (2-5 слов) уточнение для общих `event_id` (например, содержащих '_GEN') или для `event_id` 'OTHER_1848', иначе `null`.
    5.  `description`: Детальное описание события/восприятия СВОИМИ СЛОВАМИ на основе текста, объясняющее связь с революциями 1848-1849 гг.
    6.  `date_in_text`: Явная дата события из текста (не дата записи), иначе `null`.
    7.  `source_date`: Используй дату записи: "{date}".
    8.  `location`: Место события из текста. Если неясно, "Не указано".
    9.  `location_normalized`: Нормализованное место (город/страна) НА РУССКОМ. Если неясно, `null`.
    10. `brief_context`: **ОДИН-ДВА КРАТКИХ, РЕЛЕВАНТНЫХ ВНЕШНИХ ИСТОРИЧЕСКИХ ФАКТА**. **НЕ ПЕРЕСКАЗЫВАЙ дневник или `description`**. Если не нужно/неясно, "Не указано".
    11. `information_source`: Источник информации для автора из текста. Если неясно, "Не указан".
    12. `information_source_type`: ОДНО из: "Официальные источники (газеты, манифесты)", "Неофициальные сведения (слухи, разговоры в обществе)", "Личные наблюдения и опыт автора", "Информация от конкретного лица (именованный источник)", "Источник неясен/не указан". Если неясно, `null`.
    13. `confidence`: Общая уверенность ("High", "Medium", "Low") в извлеченных данных (кроме классификации).
    14. `classification_confidence`: Уверенность ("High", "Medium", "Low") в правильности `event_id`.
    15. `keywords`: Список из 3-5 ключевых слов/фраз.
    16. `text_fragment`: ТОЧНАЯ цитата (ОДНО ИЛИ НЕСКОЛЬКО ПОЛНЫХ ПРЕДЛОЖЕНИЙ) для контекста.

    **СТРОГИЕ КРИТЕРИИ ОТБОРА (ПОВТОРНО):**
    Извлекай ТОЛЬКО упоминания, которые ПРЯМО связаны с революциями 1848-1849 гг. в Европе.

    **ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА:**
    Перед включением каждого упоминания задай себе вопрос: "Можно ли это упоминание ПРЯМО связать с революционными событиями 1848-1849 гг. в Европе?" Если ответ "нет" или "возможно" - НЕ включай.

    {few_shot_examples}

    **ФОРМАТ ОТВЕТА:**
    Верни **ТОЛЬКО JSON массив объектов**. Если нет релевантных событий, верни пустой массив `[]`. JSON должен быть чистым, без каких-либо пояснений до или после.
    """


    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                reasoning_effort = "high", # для моделей Google (low, medium, high)
                #reasoning={"effort": "medium"}, # для моделе GPT с рассуждением (low, medium, high)
                max_tokens=20000, # Увеличено для более сложных моделей, если требуется больше места для ответа
                response_format={ "type": "json_object" } # Раскомментировать, если модель поддерживает и это улучшает результат
            )
            if not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                logger.error(f"Запись {entry_id}: Некорректный ответ от экстрактора. Ответ: {completion}")
                raise ValueError(f"No content in extractor response for entry_id {entry_id}")

            json_str = completion.choices[0].message.content
            try:
                if json_str.strip().startswith("```json"):
                    json_str = json_str.strip()[7:]
                    if json_str.strip().endswith("```"):
                        json_str = json_str.strip()[:-3]
                events = json.loads(json_str.strip())
            except json.JSONDecodeError as e_inner:
                logger.error(f"Внутренняя ошибка декодирования JSON для {entry_id}: {e_inner}. Строка: '{json_str[:500]}...'") # Логируем часть строки
                raise e_inner

            if not isinstance(events, list):
                logger.error(f"Экстрактор вернул не список для {entry_id}. Тип: {type(events)}. Ответ: {events}")
                raise TypeError("LLM returned non-list for event extraction")
            return events

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON для {entry_id}: {e}. Строка: '{json_str[:500] if 'json_str' in locals() else 'Не удалось получить json_str'}...'")
            if retry_count >= MAX_RETRIES -1: raise
        except Exception as e:
            retry_count += 1
            logger.warning(f"Ошибка экстракции для {entry_id} (попытка {retry_count}/{MAX_RETRIES}): {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE * (retry_count + random.uniform(0,1))) # Добавил небольшой джиттер
            else:
                logger.error(f"Превышены попытки экстракции для {entry_id}. Пропускаем.")
                return []
    return []

# -----------------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДНЕВНИКА
# -----------------------------------------------------------------------------
def process_diary():
    data = load_diary_data(DATA_PATH)
    if data.empty:
        logger.error("Не удалось загрузить данные дневника или файл пуст. Завершение работы.")
        return

    logger.info("Инициализация OpenAI клиента...")
    try:
        client = get_openai_extractor_client()
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации OpenAI клиента: {e}. Завершение работы.")
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
            logger.warning(f"Пропуск строки DataFrame с индексом {index} из-за отсутствующих данных (entry_id, date, или text is NaN).")
            continue

        try:
            current_entry_id = int(current_entry_id_raw)
        except ValueError:
            logger.warning(f"Не удалось преобразовать entry_id '{current_entry_id_raw}' в int для строки {index}. Пропуск.")
            continue

        current_text = str(current_text) # Убедимся, что текст - строка

        if current_entry_id <= last_processed_id:
            continue

        logger.info(f"Обработка записи {current_entry_id} от {current_date} ...")
        temp_file_path = os.path.join(TEMP_DIR, f"entry_{current_entry_id}.json")

        if os.path.exists(temp_file_path):
            logger.info(f"Запись {current_entry_id} уже обработана, загружаем из {temp_file_path}...")
            try:
                with open(temp_file_path, "r", encoding="utf-8") as f_temp_entry:
                    entry_events_from_temp = json.load(f_temp_entry)

                existing_event_keys = {(evt.get('entry_id'), evt.get('text_fragment', '')[:50], evt.get('event_id'))
                                       for evt in all_events if isinstance(evt, dict)}
                newly_added_count = 0
                for evt_data in entry_events_from_temp:
                    if isinstance(evt_data, dict):
                        # Убеждаемся, что entry_id и source_date из файла, если они там есть, иначе из текущей строки
                        evt_data['entry_id'] = evt_data.get('entry_id', current_entry_id)
                        evt_data['source_date'] = evt_data.get('source_date', current_date)

                        key = (evt_data.get('entry_id'),
                               evt_data.get('text_fragment', '')[:50],
                               evt_data.get('event_id'))
                        if key not in existing_event_keys:
                            try:
                                validated_event = RevolutionEvent(**ensure_default_values(evt_data))
                                all_events.append(validated_event.model_dump())
                                existing_event_keys.add(key)
                                newly_added_count += 1
                            except Exception as e_pydantic:
                                logger.error(f"Ошибка Pydantic при загрузке из temp для {current_entry_id}, событие: {evt_data}. Ошибка: {e_pydantic}")
                if newly_added_count > 0:
                    logger.info(f"Добавлено {newly_added_count} событий из temp-файла для {current_entry_id}.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки temp-файла {temp_file_path}: {e}. Обрабатываем заново.")
                # Не продолжаем, а идем на новую обработку
            else: # Если загрузка из temp прошла успешно
                try:
                    with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                        json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                    with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                        f_last_proc.write(str(current_entry_id))
                except IOError as e_save:
                    logger.error(f"Ошибка сохранения состояния после загрузки temp для {current_entry_id}: {e_save}")
                continue # Переходим к следующей записи, так как эту загрузили из temp


        try:
            extracted_events_data = extract_revolution_events(current_entry_id, current_text, current_date, client, knowledge_map_for_prompt)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"Экстрактор вернул не-словарь для {current_entry_id}: {event_item_data}.")
                    continue

                # Гарантируем, что entry_id и source_date соответствуют текущей записи
                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date

                try:
                    validated_event = RevolutionEvent(**ensure_default_values(event_item_data))
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка Pydantic для события из {current_entry_id}: {str(e_pydantic)}. Данные события: {event_item_data}")

            try:
                with open(temp_file_path, "w", encoding="utf-8") as f_entry_temp:
                    json.dump(processed_entry_events, f_entry_temp, ensure_ascii=False, indent=2)
            except IOError as e_save_entry:
                 logger.error(f"Ошибка сохранения temp-файла для {current_entry_id}: {e_save_entry}")

            all_events.extend(processed_entry_events) # Добавляем только что обработанные

            try:
                with open(TEMP_RESULTS_FILE, "w", encoding="utf-8") as f_temp_save:
                    json.dump(all_events, f_temp_save, ensure_ascii=False, indent=2)
                with open(LAST_PROCESSED_FILE, "w") as f_last_proc:
                    f_last_proc.write(str(current_entry_id))
            except IOError as e_save_main:
                 logger.error(f"Ошибка сохранения основного состояния после {current_entry_id}: {e_save_main}")

            logger.info(f"Запись {current_entry_id} обработана. Найдено {len(processed_entry_events)} событий.")

        except Exception as e_main_loop:
            logger.error(f"Критическая ошибка при обработке {current_entry_id}: {str(e_main_loop)}")
            logger.exception(f"Traceback ошибки для {current_entry_id}:")
            logger.info(f"Пропуск записи {current_entry_id} из-за ошибки...")
            try:
                with open(LAST_PROCESSED_FILE, "w") as f: # Записываем, чтобы не пытаться обработать её снова при след. запуске
                    f.write(str(current_entry_id))
            except IOError as e_save_error_state:
                 logger.error(f"Ошибка сохранения LAST_PROCESSED_FILE после ошибки для {current_entry_id}: {e_save_error_state}")
            time.sleep(5) # Небольшая пауза в случае каскадных ошибок

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
    logger.info("Запуск скрипта обработки дневника (OpenAI-совместимый API)...")
    try:
        process_diary()
    except Exception as e:
        logger.critical(f"Неперехваченная ошибка на верхнем уровне: {e}")
        logger.exception("Полный traceback неперехваченной ошибки:")
    finally:
        logger.info("Работа скрипта (OpenAI-совместимый API) завершена.")
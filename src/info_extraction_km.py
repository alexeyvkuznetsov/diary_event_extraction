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
TEMP_DIR = "temp"
LAST_PROCESSED_FILE = "last_processed.txt"
TEMP_RESULTS_FILE = "results/revolution_events_temp.json"
FINAL_RESULTS_FILE = "results/revolution_events.json"

#MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"
#MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
MODEL_NAME = "models/gemini-2.0-flash"

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
# КАРТА ЗНАНИЙ (ВСТРОЕННАЯ)
# -----------------------------------------------------------------------------
knowledge_map = """
**Универсальная Карта Знаний для Классификации Событий и Восприятий (Революции 1848-1849 гг.):**

*   **Категория: Революционные события в Европе (ID: REV1848_EUROPE)**
    *   **Подкатегория: Франция (ID: REV1848_FRA)**
        *   Февральская революция / Свержение монархии (ID: REV1848_FRA_FEB)
        *   Учреждение Второй республики (ID: REV1848_FRA_REP2)
        *   Июньское восстание рабочих (ID: REV1848_FRA_JUN)
        *   Избрание/Приход к власти Луи-Наполеона Бонапарта (ID: REV1848_FRA_NAP)
        *   Общие упоминания революции/беспорядков во Франции (ID: REV1848_FRA_GEN)
    *   **Подкатегория: Германские государства (вкл. Пруссию и Австрию вне Венгрии/Италии) (ID: REV1848_GER_AUS)**
        *   Мартовская революция (Берлин, Вена и др.) (ID: REV1848_GER_AUS_MAR)
        *   Франкфуртское национальное собрание (ID: REV1848_GER_AUS_FRA_PARL)
        *   Борьба за конституцию / Либеральные требования (ID: REV1848_GER_AUS_CONST)
        *   Национальные движения / Вопрос объединения Германии (ID: REV1848_GER_AUS_NAT)
        *   Подавление восстаний / Контрреволюция (ID: REV1848_GER_AUS_SUPP)
        *   Шлезвиг-Гольштейнский вопрос / Датско-прусская война (ID: REV1848_GER_AUS_SCH)
        *   Общие упоминания революции/беспорядков в Германии/Австрии (ID: REV1848_GER_AUS_GEN)
    *   **Подкатегория: Венгерское восстание (ID: REV1848_HUN)**
        *   Провозглашение независимости / Революционное правительство (ID: REV1848_HUN_DEC_IND)
        *   Военные действия (Австрия против Венгрии) (ID: REV1848_HUN_MIL_AUS)
        *   Российская интервенция (ID: REV1848_HUN_RUS_INT)
        *   Военные действия (Россия против Венгрии) (ID: REV1848_HUN_RUS_MIL)
        *   Капитуляция венгерской армии (Вилагош) (ID: REV1848_HUN_CAP)
        *   Личности (Кошут, Бем, Гёргей, Паскевич и др.) (ID: REV1848_HUN_FIG)
        *   Общие упоминания восстания/войны в Венгрии (ID: REV1848_HUN_GEN)
    *   **Подкатегория: Итальянские государства (ID: REV1848_ITA)**
        *   Восстания (Сицилия, Ломбардия, Венеция и др.) (ID: REV1848_ITA_UPR)
        *   Австро-сардинская война (Первая война за независимость) (ID: REV1848_ITA_AUS_SAR)
        *   Римская республика (установление, оборона, падение) (ID: REV1848_ITA_ROM_REP)
        *   Французская интервенция в Риме (ID: REV1848_ITA_FRA_INT)
        *   Личности (Карл Альберт, Мадзини, Гарибальди и др.) (ID: REV1848_ITA_FIG)
        *   Общие упоминания революции/войн в Италии (ID: REV1848_ITA_GEN)
    *   **Подкатегория: Другие регионы (Польша, Румыния и т.д.) (ID: REV1848_OTH)**
        *   Польские участники в европейских революциях (ID: REV1848_OTH_POL)
        *   Восстания/национальные движения в других регионах (ID: REV1848_OTH_MOV)
    *   **Подкатегория: Общеевропейский контекст (ID: REV1848_EUR_CONTEXT)**
        *   "Весна народов", беспорядки на Западе (общие упоминания) (ID: REV1848_EUR_CONTEXT_GEN)
        *   Распространение новостей/слухов о революциях (ID: REV1848_EUR_CONTEXT_NEWS)

*   **Категория: Реакция Российской Империи (ID: RU_REACTION_1848)**
    *   **Подкатегория: Внешняя политика и военные действия (ID: RU_REACTION_FOREIGN)**
        *   Дипломатические действия / Заявления (ID: RU_REACTION_FOREIGN_DIPLO)
        *   Манифесты о вмешательстве / войне (ID: RU_REACTION_FOREIGN_MANIFEST)
        *   Военная интервенция (с уточнением региона) (ID: RU_REACTION_FOREIGN_INTERV)
        *   Мобилизация / Передвижение войск (ID: RU_REACTION_FOREIGN_MOBIL)
        *   Рекрутские наборы (связанные с войной) (ID: RU_REACTION_FOREIGN_RECRUIT)
    *   **Подкатегория: Внутренняя политика и репрессии (ID: RU_REACTION_INTERNAL)**
        *   Усиление цензуры (ID: RU_REACTION_INTERNAL_CENS)
        *   Усиление полицейского надзора / Тайная полиция (ID: RU_REACTION_INTERNAL_POLICE)
        *   Аресты / Следствия / Суды по политическим делам (Петрашевцы и др.) (ID: RU_REACTION_INTERNAL_ARRESTS)
        *   Ссылка / Каторга по политическим мотивам (ID: RU_REACTION_INTERNAL_EXILE)
        *   Меры против "западного влияния" (указ о бритье, критика иноземщины и др.) (ID: RU_REACTION_INTERNAL_ANTIWEST)
        *   Усиление контроля над образованием / университетами (ID: RU_REACTION_INTERNAL_EDU)
    *   **Подкатегория: Общественные настроения и дискурс (ID: RU_REACTION_SOCIAL)**
        *   Страхи / Опасения перед революцией/беспорядками (ID: RU_REACTION_SOCIAL_FEAR)
        *   Патриотические настроения / Поддержка действий правительства (ID: RU_REACTION_SOCIAL_PATRIOT)
        *   Революционные / Оппозиционные / Вольнодумные настроения (ID: RU_REACTION_SOCIAL_OPPOS)
        *   Обсуждение событий / Слухи в обществе (ID: RU_REACTION_SOCIAL_DISCUSS)
        *   Анти-иноземные настроения (антипольские, антифранцузские и т.д.) (ID: RU_REACTION_SOCIAL_XENOPHOBIA)
    *   **Подкатегория: Религиозные реакции (ID: RU_REACTION_RELIGIOUS)**
        *   Официальные молебны (о начале войны, о победе и т.д.) (ID: RU_REACTION_RELIGIOUS_SERVICE)
        *   Проповеди / Религиозные тексты, связанные с событиями (ID: RU_REACTION_RELIGIOUS_SERMON)
        *   Панихиды по погибшим на войне (ID: RU_REACTION_RELIGIOUS_MEMORIAL)

*   **Категория: Идеологии и Причины (ID: IDEOLOGIES_CAUSES_1848)**
    *   **Подкатегория: Идеологические течения (ID: IDEOLOGIES_CAUSES_IDEOLOGY)**
        *   Либерализм (конституции, свободы) (ID: IDEOLOGIES_CAUSES_IDEOLOGY_LIB)
        *   Национализм (нац. единство, независимость) (ID: IDEOLOGIES_CAUSES_IDEOLOGY_NAT)
        *   Социализм / Коммунизм / Рабочий вопрос (ID: IDEOLOGIES_CAUSES_IDEOLOGY_SOC)
        *   Консерватизм / Монархизм (ID: IDEOLOGIES_CAUSES_IDEOLOGY_CONS)
    *   **Подкатегория: Социально-экономические факторы (ID: IDEOLOGIES_CAUSES_SOCEC)**
        *   Экономические кризисы / Голод / Нищета (ID: IDEOLOGIES_CAUSES_SOCEC_ECO)
        *   Социальные конфликты / Классовая борьба (ID: IDEOLOGIES_CAUSES_SOCEC_CLASS)
        *   Аграрный вопрос / Положение крестьян (ID: IDEOLOGIES_CAUSES_SOCEC_PEAS)

*   **Категория: Личный опыт и восприятие автора (ID: AUTHOR_PERCEPTION_1848)**
    *   **Подкатегория: Мнение и оценка автора (ID: AUTHOR_PERCEPTION_OPINION)**
        *   Выражение поддержки/осуждения революций/участников (ID: AUTHOR_PERCEPTION_OPINION_REV)
        *   Выражение поддержки/осуждения действий властей/конкретных государств (ID: AUTHOR_PERCEPTION_OPINION_AUTH)
        *   Размышления о политике, войне, обществе, будущем (ID: AUTHOR_PERCEPTION_OPINION_REFL)
        *   Отношение к конкретным нациям/группам (поляки, французы, венгры и т.д.) (ID: AUTHOR_PERCEPTION_OPINION_GROUPS)
    *   **Подкатегория: Эмоциональная реакция автора (ID: AUTHOR_PERCEPTION_EMOTION)**
        *   Страх, беспокойство, тревога (ID: AUTHOR_PERCEPTION_EMO_FEAR)
        *   Радость, воодушевление, надежда (ID: AUTHOR_PERCEPTION_EMO_JOY)
        *   Грусть, сочувствие, разочарование (ID: AUTHOR_PERCEPTION_EMO_SADNESS)
        *   Любопытство, интерес (ID: AUTHOR_PERCEPTION_EMO_CURIOSITY)
        *   Гнев, возмущение, ненависть (ID: AUTHOR_PERCEPTION_EMO_ANGER)
        *   Общая эмоциональная реакция/сильное впечатление (неуточненная эмоция) (ID: AUTHOR_PERCEPTION_EMO_GENERAL)
    *   **Подкатегория: Влияние событий на автора/окружение (ID: AUTHOR_PERCEPTION_IMPACT)**
        *   Прямое влияние событий на жизнь/планы автора (ID: AUTHOR_PERCEPTION_IMPACT_SELF)
        *   Влияние событий на знакомых/родственников автора (ID: AUTHOR_PERCEPTION_IMPACT_ACQ)

*   **Категория: Другое / Не классифицировано (ID: OTHER_1848)** (Только если упоминание связано с революциями, но не подходит под карту)
"""

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

def load_diary_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ И ВЕРИФИКАЦИИ ДАННЫХ
# -----------------------------------------------------------------------------
def extract_revolution_events(entry_id: int, text: str, date: str, extractor_model, current_knowledge_map_str: str) -> List[Dict[str, Any]]:
    manage_api_rate_limit()
    # current_knowledge_map_str здесь будет глобальной переменной `knowledge_map`
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

    Используй Карту Знаний для event_id и event_name:
    {current_knowledge_map_str}

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
    # current_knowledge_map_str здесь будет глобальной переменной `knowledge_map`
    if not isinstance(event_data, dict):
        logger.error(f"verify_event получил не словарь: {type(event_data)}.")
        return event_data

    event_data_to_verify = ensure_default_values(event_data.copy())
    original_event_id = event_data_to_verify.get('event_id')
    original_text_fragment = event_data_to_verify.get('text_fragment')

    prompt = f"""
    Проверь и при необходимости исправь следующую информацию о событии, извлеченную из дневника:
    ```json
    {json.dumps(event_data_to_verify, ensure_ascii=False, indent=2)}
    ```
    Полный текст дневниковой записи для контекста: "{full_text}"
    Используй Карту Знаний для проверки `event_id` и `event_name`: {current_knowledge_map_str}
    Задачи:
    1. Критически оцени `event_id` и `classification_confidence`.
    2. Убедись, что `description` точно отражает `text_fragment`.
    3. Проверь и скорректируй поля: `event_subtype_custom`, `location_normalized`, `information_source_type` (используй ТОЛЬКО значения из списка).
    4. Убедись, что в `brief_context`  указан **конкретный исторический факт (1-2 предложения)**, а не мнение или общее рассуждение. Если нет, исправь или установи 'Не указано'..
    5. Убедись, что `event_name` соответствует названию из Карты Знаний для данного `event_id` (без префиксов типа 'Событие:'), или является кастомным для 'OTHER_1848'/null.
    6. Заполни `location` или `information_source`, если возможно из текста.
    7. Проверь `text_fragment`: он должен быть полным предложением/ями для контекста.
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
    # Глобальная переменная knowledge_map используется напрямую
    # в extract_revolution_events и verify_event

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

    all_events: List[Dict[str, Any]] = []
    if not os.path.exists(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
            logger.info(f"Создана директория {TEMP_DIR}")
        except OSError as e:
            logger.error(f"Не удалось создать директорию {TEMP_DIR}: {e}. Завершение работы.")
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

        if current_entry_id <= last_processed_id:
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
                                validated_event = RevolutionEvent(**ensure_default_values(evt_data))
                                all_events.append(validated_event.model_dump())
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
            # Передаем глобальную knowledge_map в функции
            extracted_events_data = extract_revolution_events(current_entry_id, current_text, current_date, extractor_model, knowledge_map)
            processed_entry_events = []

            for event_item_data in extracted_events_data:
                if not isinstance(event_item_data, dict):
                    logger.warning(f"Экстрактор вернул не-словарь для {current_entry_id}: {event_item_data}. Пропускаем.")
                    continue

                event_item_data['entry_id'] = current_entry_id
                event_item_data['source_date'] = current_date
                verified_event_item_data = verify_event(event_item_data, verifier_model, current_text, knowledge_map)

                if not isinstance(verified_event_item_data, dict):
                    logger.warning(f"Верификатор вернул не-словарь для {current_entry_id}: {verified_event_item_data}. Пропускаем.")
                    continue

                verified_event_item_data['entry_id'] = current_entry_id
                verified_event_item_data['source_date'] = current_date

                try:
                    validated_event = RevolutionEvent(**verified_event_item_data)
                    processed_entry_events.append(validated_event.model_dump())
                except Exception as e_pydantic:
                    logger.error(f"Ошибка валидации Pydantic для события из {current_entry_id}: {e_pydantic}")
                    logger.debug(f"Данные события (Pydantic ошибка): {verified_event_item_data}")
                    try:
                        deep_fixed_data = ensure_default_values(verified_event_item_data.copy())
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
    except IOError as e:
        logger.error(f"Не удалось сохранить финальный файл {FINAL_RESULTS_FILE}: {e}")

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
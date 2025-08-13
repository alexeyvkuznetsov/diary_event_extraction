# СОЗДАЕТ ФАЙЛ ДЛЯ ДАШБОРДА

import json
from datetime import datetime

def process_diary_data(input_filename='revolution_events.json', output_filename='dashboard_data.json'):
    """
    Читает "чистый" JSON-файл с событиями, рассчитывает статистику,
    формирует данные для фильтров и сохраняет всё в новом JSON-файле,
    совместимом с дашбордом.
    """
    print(f"Загрузка данных из '{input_filename}'...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            events = json.load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{input_filename}' не найден. Убедитесь, что он находится в той же папке.")
        return
    except json.JSONDecodeError:
        print(f"ОШИБКА: Файл '{input_filename}' содержит некорректный JSON.")
        return

    print(f"Найдено {len(events)} событий. Начинаю обработку...")

    # --- Инициализация структур для сбора данных ---

    # Для фильтров (уникальные значения)
    locations = set()
    source_types = set()
    confidence_levels = set()

    # Для статистики (счетчики)
    months_distribution = {}
    prefixes_distribution = {}
    locations_distribution = {}
    source_types_distribution = {}
    confidence_distribution = {}

    # Для определения временного диапазона
    all_dates = []

    # --- Предопределенные названия для категорий ---
    category_names = {
        'REV1848_': 'Революции 1848-49',
        'RU_': 'Российские реакции',
        'AUTHOR_': 'Авторские восприятия',
        'IDEOLOGIES_': 'Идеологии и причины',
        'OTHER_': 'Прочее'
    }

    month_names_map = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь",
        7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
    }

    # --- Основной цикл обработки событий ---
    for event in events:
        # Дата
        if event.get('source_date'):
            date_obj = datetime.strptime(event['source_date'], '%Y-%m-%d')
            all_dates.append(date_obj)
            month_key = date_obj.strftime('%Y-%m')
            months_distribution[month_key] = months_distribution.get(month_key, 0) + 1

        # Категория (префикс)
        if event.get('event_id'):
            prefix = event['event_id'].split('_')[0] + '_'
            prefixes_distribution[prefix] = prefixes_distribution.get(prefix, 0) + 1

        # Локация
        if event.get('location_normalized'):
            loc = event['location_normalized']
            locations.add(loc)
            locations_distribution[loc] = locations_distribution.get(loc, 0) + 1

        # Тип источника
        if event.get('information_source_type'):
            src_type = event['information_source_type']
            source_types.add(src_type)
            source_types_distribution[src_type] = source_types_distribution.get(src_type, 0) + 1

        # Уровень достоверности
        if event.get('confidence'):
            conf = event['confidence']
            confidence_levels.add(conf)
            confidence_distribution[conf] = confidence_distribution.get(conf, 0) + 1

    # --- Финализация и сборка итогового объекта ---

    # Формируем данные для фильтров
    months_filter = {}
    if all_dates:
        sorted_month_keys = sorted(list(months_distribution.keys()))
        for key in sorted_month_keys:
            year, month = map(int, key.split('-'))
            months_filter[key] = f"{month_names_map[month]} {year}"

    filters = {
        "months": months_filter,
        "event_prefixes": category_names,
        "locations": sorted(list(locations)),
        "source_types": sorted(list(source_types)),
        "confidence_levels": sorted(list(confidence_levels), key=lambda x: ['High', 'Medium', 'Low'].index(x) if x in ['High', 'Medium', 'Low'] else 99)
    }

    # Формируем статистику
    statistics = {
        "total_events": len(events),
        "date_range": {
            "start": min(all_dates).strftime('%Y-%m-%d') if all_dates else None,
            "end": max(all_dates).strftime('%Y-%m-%d') if all_dates else None
        },
        "months_distribution": months_distribution,
        "prefixes_distribution": {k: v for k, v in prefixes_distribution.items() if k in category_names},
        "locations_distribution": locations_distribution,
        "source_types_distribution": source_types_distribution,
        "confidence_distribution": confidence_distribution
    }

    # Собираем финальный JSON
    output_data = {
        "events": events,
        "filters": filters,
        "statistics": statistics
    }

    # --- Сохранение результата в файл ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nОбработка завершена. Результат сохранен в файл '{output_filename}'.")

# --- Запуск скрипта ---
if __name__ == "__main__":
    process_diary_data()
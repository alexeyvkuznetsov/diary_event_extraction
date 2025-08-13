import csv
import re
import os

def fix_multiline_csv(input_filepath, output_filepath):
    """
    Читает некорректно отформатированный CSV-файл с многострочными полями,
    восстанавливает целостность строк и записывает новый, правильный CSV-файл.

    Args:
        input_filepath (str): Путь к исходному "сломанному" CSV-файлу.
        output_filepath (str): Путь для сохранения исправленного CSV-файла.
    """
    print(f"Начинаю обработку файла: '{input_filepath}'...")

    # Регулярное выражение для определения начала новой логической строки.
    # Ищет строки, начинающиеся с цифр, за которыми следует точка с запятой.
    start_of_row_pattern = re.compile(r'^\d+;')

    corrected_rows = []
    current_row_lines = []
    header = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            # Читаем заголовок отдельно
            header_line = infile.readline().strip()
            if header_line:
                header = [h.strip() for h in header_line.split(';')]
            else:
                print("Ошибка: Входной файл пуст или не содержит заголовка.")
                return

            # Обрабатываем остальные строки
            for line in infile:
                # Проверяем, похожа ли строка на начало новой записи
                if start_of_row_pattern.match(line):
                    # Если буфер предыдущей строки не пуст, обрабатываем его
                    if current_row_lines:
                        full_row_str = "".join(current_row_lines).strip()
                        # Разделяем строку только по первым двум разделителям
                        parts = full_row_str.split(';', 2)
                        if len(parts) == 3:
                            corrected_rows.append(parts)

                    # Начинаем новый буфер с текущей строки
                    current_row_lines = [line]
                else:
                    # Если это не новая запись, то это продолжение текста
                    current_row_lines.append(line)

            # Не забываем обработать последнюю запись, оставшуюся в буфере
            if current_row_lines:
                full_row_str = "".join(current_row_lines).strip()
                parts = full_row_str.split(';', 2)
                if len(parts) == 3:
                    corrected_rows.append(parts)

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_filepath}' не найден.")
        return
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return

    # Записываем исправленные данные в новый файл с помощью модуля csv
    print(f"Обнаружено {len(corrected_rows)} логических строк. Идет запись в '{output_filepath}'...")
    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            # Используем writer из модуля csv для корректной обработки кавычек
            writer = csv.writer(outfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)

            # Записываем заголовок
            if header:
                writer.writerow(header)

            # Записываем обработанные строки
            for row_parts in corrected_rows:
                # Очищаем каждую часть от лишних пробелов и кавычек
                # (csv.writer сам добавит кавычки, где это необходимо)
                cleaned_row = [part.strip().strip('"') for part in row_parts]
                writer.writerow(cleaned_row)

    except Exception as e:
        print(f"Произошла ошибка при записи файла: {e}")
        return

    print("-" * 25)
    print("Файл успешно исправлен.")
    print(f"Корректные данные сохранены в: '{os.path.abspath(output_filepath)}'")
    print("-" * 25)

# --- ИНСТРУКЦИЯ ПО ПРИМЕНЕНИЮ ---
# 1. Сохраните этот код в файл с именем, например, `fix_csv.py`.
# 2. Поместите ваш файл `kapustin.csv` в ту же папку, где лежит скрипт.
# 3. Запустите скрипт из терминала (командной строки) командой: python fix_csv.py

if __name__ == "__main__":
    input_filename = 'kapustin.csv'
    output_filename = 'kapustin_corrected.csv'

    if os.path.exists(input_filename):
        fix_multiline_csv(input_filename, output_filename)
    else:
        print(f"Ошибка: Не могу найти файл '{input_filename}'. Убедитесь, что он находится в той же папке, что и скрипт.")
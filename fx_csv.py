import csv
import re
import os

def fix_and_prepare_for_excel(input_filepath, output_filepath):
    """
    Читает некорректно отформатированный CSV, восстанавливает строки
    и сохраняет его в формате, максимально совместимом с Microsoft Excel
    (разделитель-запятая, кодировка UTF-8 с BOM).

    Args:
        input_filepath (str): Путь к исходному "сломанному" CSV-файлу.
        output_filepath (str): Путь для сохранения исправленного CSV-файла.
    """
    print(f"Начинаю обработку файла: '{input_filepath}'...")

    start_of_row_pattern = re.compile(r'^\d+;')

    corrected_rows = []
    current_row_lines = []
    header = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            header_line = infile.readline().strip()
            # Важно: меняем разделитель и в заголовке
            if header_line:
                header = [h.strip() for h in header_line.split(';')]
            else:
                print("Ошибка: Входной файл пуст или не содержит заголовка.")
                return

            for line in infile:
                if start_of_row_pattern.match(line):
                    if current_row_lines:
                        full_row_str = "".join(current_row_lines).strip()
                        parts = full_row_str.split(';', 2)
                        if len(parts) == 3:
                            corrected_rows.append(parts)
                    current_row_lines = [line]
                else:
                    current_row_lines.append(line)

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

    print(f"Обнаружено {len(corrected_rows)} строк. Идет запись в '{output_filepath}'...")
    try:
        # --- КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ ЗДЕСЬ ---
        with open(output_filepath, 'w', newline='', encoding='utf-8-sig') as outfile:
            # 1. Меняем разделитель на запятую
            writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
            #    quoting=csv.QUOTE_ALL заставит обернуть все поля в кавычки, что повышает надежность.
            # 2. Кодировка 'utf-8-sig' автоматически добавляет BOM-маркер.

            if header:
                writer.writerow(header)

            for row_parts in corrected_rows:
                cleaned_row = [part.strip().strip('"') for part in row_parts]
                writer.writerow(cleaned_row)

    except Exception as e:
        print(f"Произошла ошибка при записи файла: {e}")
        return

    print("-" * 25)
    print("Файл успешно исправлен и подготовлен для Excel.")
    print(f"Новый файл сохранен как: '{os.path.abspath(output_filepath)}'")
    print("-" * 25)

# --- ИНСТРУКЦИЯ ПО ПРИМЕНЕНИЮ ---
if __name__ == "__main__":
    # Используем исходный "сломанный" файл
    input_filename = 'kapustin.csv'
    # Создаем новый файл с другим именем, чтобы не перезаписать предыдущий результат
    output_filename = 'kapustin_excel_compatible.csv'

    if os.path.exists(input_filename):
        fix_and_prepare_for_excel(input_filename, output_filename)
    else:
        print(f"Ошибка: Не могу найти файл '{input_filename}'. Убедитесь, что он в той же папке.")
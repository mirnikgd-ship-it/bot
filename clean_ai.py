import psycopg2
import os
import json
import time
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
# ==========================================
# 1. НАСТРОЙКИ ПОДКЛЮЧЕНИЯ
# ==========================================
DB_HOST = os.getenv('DB_HOST', '192.168.133.20')
DB_NAME = os.getenv('DB_NAME', 'recipes_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '123Secret_a') # <-- ВАШ ПАРОЛЬ

# Ключ авторизации GigaChat (получается в личном кабинете)
GIGACHAT_CREDENTIALS="MDFhMTVkZjEtZDI1ZC03YzJkLWJhNzQtMzYyMTdkNDk1MTA0OjVjMWUwZjc0LWI0MzAtNGUxOC05ODM1LWRkMTJlZjNiMDM3ZQ=="
BATCH_SIZE = 10  # Уменьшили, чтобы не упираться в лимит за раз

# ==========================================
# 2. ОЧИСТКА ЧЕРЕЗ GIGACHAT (с retry)
# ==========================================
def clean_ingredients_with_gigachat(raw_ingredients):
    # ЖЕСТКИЙ ПРОМПТ С ПРИМЕРАМИ
    prompt = f"""
Ты — безжалостный парсер кулинарных данных. Твоя задача — извлечь САМЫЙ БАЗОВЫЙ продукт из строки.
Ты должен игнорировать:
- Бренды и производителей (ТМ, ООО, названия на латинице, имена собственные).
- Состояния и способы нарезки (вареный, жареный, нарезанный, молотый).
- Количество (гр, кг, шт, стаканы, ложки, щепотки).
- Маркетинговые фразы ("на ваш выбор", "для кляра", "по вкусу").
- Уточнения в скобках.

Если в строке несколько разных продуктов (например, "соль и перец"), оставь только ПЕРВЫЙ.
Если строка — это просто мусор или категория блюда (например, "Салаты из сыра", "Украшения из фруктов"), верни "УДАЛИТЬ".

Вот примеры правильной очистки:
Вход: "Индейка ТМ ИНДИЛАЙТ голень варено" -> Выход: "индейка"
Вход: "Индейка ТМ*Индилайт* Sabroso Roso" -> Выход: "индейка"
Вход: "Кус-кус крупа пшеничная продукции "Мистраль"" -> Выход: "кускус"
Вход: "Яйцо куриное 1 в фарш\n3 для кляра" -> Выход: "яйцо куриное"
Вход: "Специи соль и черный перец, по вкусу" -> Выход: "соль"
Вход: "Сыр мягкий klassischer" -> Выход: "сыр мягкий"
Вход: "Салаты из сыра Салаты из бобовых" -> Выход: "УДАЛИТЬ"

Твоя очередь. Верни ответ СТРОГО в формате JSON-массива строк. Порядок должен строго совпадать с исходным списком.
Пример ответа: ["сахар", "цедра апельсина", "УДАЛИТЬ", "приправа"]

Список для очистки:
{json.dumps(raw_ingredients, ensure_ascii=False)}
"""
    
    try:
        # ИСПОЛЬЗУЕМ БОЛЕЕ УМНУЮ МОДЕЛЬ, ЕСЛИ ОНА ДОСТУПНА
        with GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            base_url="https://api.giga.chat/v1",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2-Max",  # <--- РЕКОМЕНДУЮ MAX или PRO, вместо просто Lite
            verify_ssl_certs=False,
            max_retries=3,
            retry_backoff_factor=1.0,
            retry_on_status_codes=(429, 500, 502, 503, 504)
        ) as client:
            
            # Используем Chat API для более точного контроля
            response = client.chat(
                Chat(
                    messages=[Messages(role=MessagesRole.USER, content=prompt)],
                    temperature=0.0  # 0.0 = никакой фантазии, только факты
                )
            )
            
            # В Gigachat Chat API ответ лежит в .choices[0].message.content
            result_text = response.choices[0].message.content
            
            # Очистка от markdown-оберток
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json\n", "").replace("```", "").strip()
            elif result_text.startswith("```"):
                result_text = result_text.replace("```\n", "").replace("```", "").strip()

            data = json.loads(result_text)
            
            if isinstance(data, list) and len(data) == len(raw_ingredients):
                return data
            else:
                print(f"  [!] Ошибка формата ответа (длина не совпадает или не список): {result_text[:100]}")
                return None
                
    except Exception as e:
        print(f"  [x] Ошибка GigaChat: {e}")
        return None

# ==========================================
# 3. ОБНОВЛЕНИЕ БАЗЫ (с паузами)
# ==========================================
def process_db():
    conn = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name FROM ingredients 
        WHERE length(name) > 15 
           OR name ~ '[0-9]' 
           OR name LIKE '%,%'
           OR name LIKE '%(%'
    """)
    rows = cursor.fetchall()
    print(f"Найдено {len(rows)} ингредиентов для очистки.")
    
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        raw_names = [r[1] for r in batch]
        
        print(f"Обработка пачки {i//BATCH_SIZE + 1} (из { (len(rows)-1)//BATCH_SIZE + 1 })...")
        
        clean_names = clean_ingredients_with_gigachat(raw_names)
        
        if not clean_names or len(clean_names) != len(raw_names):
            print("  [!] Ошибка в ответе GigaChat, пропускаем пачку.")
            time.sleep(5)  # Если не получилось — делаем паузу подольше
            continue
            
        for (old_id, old_name), new_name in zip(batch, clean_names):
            new_name = new_name.strip().lower()
            
            if new_name == "удалить" or not new_name:
                cursor.execute("DELETE FROM ingredients WHERE id = %s", (old_id,))
                print(f"  [-] Удалено: {old_name}")
                continue
                
            if new_name == old_name.lower():
                continue
                
            cursor.execute("SELECT id FROM ingredients WHERE name = %s", (new_name,))
            existing = cursor.fetchone()
            
            if existing:
                existing_id = existing[0]
                cursor.execute("""
                    INSERT INTO recipe_ingredients (recipe_id, ingredient_id, amount, full_text)
                    SELECT recipe_id, %s, amount, full_text FROM recipe_ingredients WHERE ingredient_id = %s
                    ON CONFLICT DO NOTHING
                """, (existing_id, old_id))
                cursor.execute("DELETE FROM ingredients WHERE id = %s", (old_id,))
                print(f"  [+] Объединено: '{old_name}' -> '{new_name}'")
            else:
                cursor.execute("UPDATE ingredients SET name = %s WHERE id = %s", (new_name, old_id))
                print(f"  [+] Очищено: '{old_name}' -> '{new_name}'")
                
        conn.commit()
        
        # ПАУЗА МЕЖДУ ПАЧКАМИ — чтобы не получить 429
        time.sleep(2.5)
        
    conn.close()
    print("Готово! База данных очищена.")

if __name__ == '__main__':
    process_db()
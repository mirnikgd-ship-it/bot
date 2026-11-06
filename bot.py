import psycopg2
import os
import re
import base64
import traceback

# ==========================================
# 1. НАСТРОЙКИ ПОДКЛЮЧЕНИЯ К БД
# ==========================================
# Если вы запускаете бота на компьютере, а база на сервере, 
# укажите здесь IP-адрес сервера вместо 'localhost'
DB_HOST = os.getenv('DB_HOST', '192.168.133.20')
DB_NAME = os.getenv('DB_NAME', 'recipes_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '123Secret_a') # <-- УКАЖИТЕ ВАШ ПАРОЛЬ

# Коэффициенты для формулы k*d + m*f
K_COEF = 1.0
M_COEF = -2.0  # Отрицательный, чтобы частые ингредиенты были "лучше"

# ==========================================
# 2. ПОИСК РЕЦЕПТОВ (С формулой k*d + m*f)
# ==========================================
def find_recipes_smart(user_input_text):
    """Ищет рецепты, используя формулу k*d + m*f"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
        )
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
        conn.commit()
        
        clean_input = ''.join(c for c in user_input_text if c.isprintable())
        
        query = r"""
        WITH user_words AS (
            -- 1. Разбиваем запрос пользователя на слова
            SELECT DISTINCT lower(regexp_replace(word, '[^a-zа-яё]', '', 'g')) AS word
            FROM regexp_split_to_table(%s::text, '\s+') AS word
            WHERE length(regexp_replace(word, '[^a-zа-яё]', '', 'g')) >= 3
        ),
        db_words AS (
            -- 2. Разбиваем названия ингредиентов в БД на слова
            SELECT id, name, lower(regexp_replace(word, '[^a-zа-яё]', '', 'g')) AS db_word
            FROM ingredients, regexp_split_to_table(name, '\s+') AS word
            WHERE length(name) < 50
        ),
        -- 3. Считаем частоту каждого ингредиента в рецептах (freq)
        ingredient_freq AS (
            SELECT i.id, COALESCE(COUNT(ri.recipe_id), 0) as freq
            FROM ingredients i
            LEFT JOIN recipe_ingredients ri ON i.id = ri.ingredient_id
            GROUP BY i.id
        ),
        -- 4. Находим минимальную и максимальную частоту
        min_max_freq AS (
            SELECT MIN(freq) as min_f, MAX(freq) as max_f FROM ingredient_freq
        ),
        -- 5. Нормализуем частоту (f): (freq - min) / (max - min)
        normalized_freq AS (
            SELECT 
                f.id,
                CASE 
                    WHEN m.max_f = m.min_f THEN 0.0 
                    ELSE (f.freq - m.min_f)::numeric / (m.max_f - m.min_f) 
                END as f_val
            FROM ingredient_freq f
            CROSS JOIN min_max_freq m
        ),
        -- 6. Считаем расстояние Левенштейна (d) и итоговый скор
        distances AS (
            SELECT 
                dw.id, 
                dw.name, 
                levenshtein(dw.db_word, uw.word) AS d_val,
                nf.f_val
            FROM db_words dw
            JOIN user_words uw ON length(dw.db_word) >= 3
            JOIN normalized_freq nf ON dw.id = nf.id
            WHERE levenshtein(dw.db_word, uw.word) <= 3 -- Отсекаем совсем непохожие
        ),
        scored AS (
            -- ВЫЧИСЛЯЕМ ФОРМУЛУ: (k * d) + (m * f)
            SELECT 
                id,
                name,
                MIN((%s * d_val) + (%s * f_val)) AS best_score
            FROM distances
            GROUP BY id, name
        ),
        -- 7. Берем топ-20 ингредиентов с наименьшим скором
        best_matches AS (
            SELECT id, name, best_score
            FROM scored
            ORDER BY best_score ASC
            LIMIT 20
        )
        -- 8. Выводим рецепты, где встречаются эти ингредиенты
        SELECT 
            encode(textsend(r.title), 'base64') as title_b64, 
            r.url, 
            encode(textsend(string_agg(i.name, ', ')), 'base64') as matched_ings_b64
        FROM recipes r
        JOIN recipe_ingredients ri ON r.id = ri.recipe_id
        JOIN ingredients i ON ri.ingredient_id = i.id
        JOIN best_matches bm ON i.id = bm.id
        GROUP BY r.id, r.title, r.url
        ORDER BY COUNT(i.id) DESC
        LIMIT 5;
        """
        
        # Передаем текст пользователя и коэффициенты K и M
        cursor.execute(query, (clean_input, K_COEF, M_COEF))
        raw_results = cursor.fetchall()
        conn.close()
        
        results = []
        for row in raw_results:
            title_b64, url, ings_b64 = row
            try:
                title = base64.b64decode(title_b64).decode('utf-8', errors='replace')
                matched_ings = base64.b64decode(ings_b64).decode('utf-8', errors='replace')
                results.append((title, url, matched_ings))
            except Exception:
                pass
            
        return results
    except Exception as e:
        print(f"\n--- ОШИБКА ---\n{e}\n-------------")
        return []

# ==========================================
# 3. ИНТЕРФЕЙС
# ==========================================
def main():
    print("\n" + "="*50)
    print(" 🍳 БОТ (Формула k*d + m*f)")
    print("="*50)
    
    while True:
        user_input = input("Ваши продукты: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']: break
        if not user_input: continue
            
        print(f"🔍 Ищем лучшие совпадения для: '{user_input}'...")
        
        results = find_recipes_smart(user_input)
        
        if not results:
            print("😔 Ничего не нашлось.\n")
            continue
            
        print(f"\n✅ Найдено рецептов: {len(results)}\n")
        
        for i, r in enumerate(results):
            title, url, matched_ings = r
            print(f"{i+1}. {title}")
            print(f"   Ингредиенты: {matched_ings}")
            print(f"   {url}")
            print("-" * 50)
            
        print("\n")

if __name__ == '__main__':
    main()
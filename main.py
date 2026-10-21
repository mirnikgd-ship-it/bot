import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
import re
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# === ИМПОРТЫ ДЛЯ NLP ===
from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger, Doc, Segmenter
from nltk.stem.snowball import SnowballStemmer

# ==========================================
# 0. ИНИЦИАЛИЗАЦИЯ NLP + КЭШ
# ==========================================
print("Загрузка NLP моделей...")
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stemmer = SnowballStemmer("russian")

# КЭШ: словарь для хранения уже обработанных слов {имя: (лемма, стемм)}
nlp_cache = {}
cache_lock = threading.Lock() # Защита кэша при многопоточности

def process_ingredient_name(text):
    """Возвращает (лемма, стемм). Использует кэш для ускорения."""
    if text in nlp_cache:
        return nlp_cache[text]
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma:
            lemmas.append(token.lemma)
        else:
            lemmas.append(token.text.lower())
            
    lemma_result = " ".join(lemmas).replace('(', '').replace(')', '').strip()
    stem_result = " ".join([stemmer.stem(word) for word in lemmas]).replace('(', '').replace(')', '').strip()
    
    with cache_lock:
        nlp_cache[text] = (lemma_result, stem_result)
        
    return lemma_result, stem_result

# ==========================================
# 1. БАЗА ДАННЫХ
# ==========================================
def init_db(db_name='recipes.db'):
    conn = sqlite3.connect(db_name, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            lemma TEXT,
            stem TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            recipe_id INTEGER,
            ingredient_id INTEGER,
            amount TEXT,
            full_text TEXT,
            PRIMARY KEY (recipe_id, ingredient_id),
            FOREIGN KEY (recipe_id) REFERENCES recipes (id) ON DELETE CASCADE,
            FOREIGN KEY (ingredient_id) REFERENCES ingredients (id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    return conn

# ==========================================
# 2. ОЧИСТКА И РАЗДЕЛЕНИЕ ТЕКСТА
# ==========================================
def split_ingredient(text):
    """Очищает от скобок (включая одиночные) и делит на название и количество"""
    text_cleaned = re.sub(r'\(.*?\)', '', text)
    text_cleaned = re.sub(r'[\(\)]+', '', text_cleaned).strip()
    parts = re.split(r'\s+[—–-]\s+', text_cleaned, maxsplit=1)
    
    if len(parts) == 2:
        name = parts[0].strip()
        amount = parts[1].strip()
        name = re.sub(r'[\(\)]+$', '', name).strip()
        name = re.sub(r'^[\(\)]+', '', name).strip()
        if not name: return text.strip(), ""
        return name, amount
    else:
        name = re.sub(r'[\(\)]+$', '', text_cleaned).strip()
        name = re.sub(r'^[\(\)]+', '', name).strip()
        return name, ""

# ==========================================
# 3. ПАРСИНГ ОДНОГО РЕЦЕПТА
# ==========================================
def parse_povarenok_recipe(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "Без названия"
        
        ingredients = []
        ing_container = soup.find('div', class_='ingredients-bl')
        if ing_container:
            for li in ing_container.find_all('li'):
                text = li.get_text(separator=' ', strip=True)
                if text: ingredients.append(text)
        else:
            ing_header = soup.find(lambda tag: tag.name in ['h2', 'h3', 'div'] and 'Ингредиенты' in tag.get_text())
            if ing_header:
                ing_list = ing_header.find_next('ul')
                if ing_list:
                    for li in ing_list.find_all('li'):
                        text = li.get_text(separator=' ', strip=True)
                        if text and len(text) < 150: ingredients.append(text)
                                
        clean_ingredients = [ing for ing in ingredients if not any(w in ing.lower() for w in ['комментарий', 'отзыв'])]
        return {'url': url, 'title': title, 'ingredients': clean_ingredients}
    except Exception:
        return None

# ==========================================
# 4. СОХРАНЕНИЕ В БД (Пакетное)
# ==========================================
def save_to_db(conn, recipe_data):
    if not recipe_data or not recipe_data['ingredients']: return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO recipes (url, title) VALUES (?, ?)", (recipe_data['url'], recipe_data['title']))
        if cursor.rowcount != 1: return False
        
        recipe_id = cursor.lastrowid
        recipe_ingredients_to_insert = []
        
        for ing in recipe_data['ingredients']:
            name, amount = split_ingredient(ing)
            if not name: continue
            name = name.replace('(', '').replace(')', '').strip()
            if not name: continue
            
            cursor.execute("SELECT id FROM ingredients WHERE name = ?", (name,))
            row = cursor.fetchone()
            
            if row:
                ingredient_id = row[0]
            else:
                lemma, stem = process_ingredient_name(name)
                cursor.execute("INSERT INTO ingredients (name, lemma, stem) VALUES (?, ?, ?)", (name, lemma, stem))
                ingredient_id = cursor.lastrowid
            
            recipe_ingredients_to_insert.append((recipe_id, ingredient_id, amount, ing))
            
        if recipe_ingredients_to_insert:
            cursor.executemany("INSERT OR IGNORE INTO recipe_ingredients (recipe_id, ingredient_id, amount, full_text) VALUES (?, ?, ?, ?)", recipe_ingredients_to_insert)
            
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Ошибка БД: {e}")
        conn.rollback()
        return False

# ==========================================
# 5. УСКОРЕННЫЙ СБОР ССЫЛОК
# ==========================================
def get_links_from_page(page_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        res = requests.get(page_url, headers=headers, timeout=10)
        if res.status_code != 200: return []
        soup = BeautifulSoup(res.text, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            if '/recipes/show/' in a['href']:
                clean_url = urljoin('https://www.povarenok.ru', a['href']).split('#')[0].rstrip('/') + '/'
                links.add(clean_url)
        return list(links)
    except:
        return []

def get_all_site_links():
    """Собирает ссылки со всех страниц пагинации сайта параллельно"""
    base_url = 'https://www.povarenok.ru/recipes/'
    all_links = set()
    page = 1
    stop_scanning = False
    
    while not stop_scanning:
        # Берем пачку из 10 страниц для параллельного сканирования
        urls_to_scan = []
        for _ in range(10):
            urls_to_scan.append(base_url if page == 1 else f"{base_url}~{page}/")
            page += 1
            
        print(f"  Сканируем пачку страниц (до {page-1})...")
        found_in_batch = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(get_links_from_page, url): url for url in urls_to_scan}
            for future in as_completed(futures):
                page_links = future.result()
                if not page_links: # Если страница пустая или 404, значит дошли до конца
                    stop_scanning = True 
                else:
                    for link in page_links:
                        if link not in all_links:
                            all_links.add(link)
                            found_in_batch += 1
                            
        print(f"    +{found_in_batch} новых ссылок (всего: {len(all_links)})")
        if found_in_batch == 0: # Если 10 страниц подряд не дали ничего нового, останавливаемся
            stop_scanning = True
            
        time.sleep(random.uniform(1, 2))
        
    return list(all_links)

# ==========================================
# 6. ГЛАВНЫЙ ЗАПУСК (СБОР ВСЕГО САЙТА)
# ==========================================
if __name__ == '__main__':
    conn = init_db()
    
    print("Начинаем сбор ВСЕХ ссылок на рецепты с сайта...")
    all_urls = get_all_site_links()
    print(f"\nСБОР ССЫЛОК ЗАВЕРШЕН. Всего найдено рецептов: {len(all_urls)}")
    
    # Проверяем, что уже есть в базе, чтобы не парсить дважды
    cursor = conn.cursor()
    new_urls = []
    for url in all_urls:
        cursor.execute("SELECT id FROM recipes WHERE url = ?", (url,))
        if not cursor.fetchone():
            new_urls.append(url)
            
    print(f"Новых для парсинга: {len(new_urls)}")
    print("Начинаем скачивать и парсить каждый рецепт...\n")
    
    total_saved = 0
    for i, url in enumerate(new_urls):
        print(f"[{i+1}/{len(new_urls)}] {url}")
        data = parse_povarenok_recipe(url)
        
        if data and save_to_db(conn, data):
            total_saved += 1
            print(f"  + Сохранено: {data['title']}")
            
        # Пауза между рецептами. Если получите бан — увеличьте это значение.
        time.sleep(random.uniform(1, 2.5))
        
    conn.close()
    print(f"\n=== РАБОТА ЗАВЕРШЕНА. Сохранено новых рецептов: {total_saved} ===")
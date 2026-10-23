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
# 0. НАСТРОЙКИ СКОРОСТИ
# ==========================================
MAX_WORKERS = 12        # Сколько рецептов качать ОДНОВРЕМЕННО (не ставьте больше 15, будет бан)
BATCH_SAVE_SIZE = 50    # Сохранять в базу пачками по 50 рецептов
SLEEP_BETWEEN_BATCHES = 0.5 # Пауза после каждой пачки

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ NLP + КЭШ
# ==========================================
print("Загрузка NLP моделей...")
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stemmer = SnowballStemmer("russian")

nlp_cache = {}
cache_lock = threading.Lock()

def process_ingredient_name(text):
    if text in nlp_cache: return nlp_cache[text]
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma if token.lemma else token.text.lower())
            
    lemma_result = " ".join(lemmas).replace('(', '').replace(')', '').strip()
    stem_result = " ".join([stemmer.stem(word) for word in lemmas]).replace('(', '').replace(')', '').strip()
    
    with cache_lock:
        nlp_cache[text] = (lemma_result, stem_result)
    return lemma_result, stem_result

# ==========================================
# 2. БАЗА ДАННЫХ
# ==========================================
def init_db(db_name='recipes.db'):
    conn = sqlite3.connect(db_name, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS recipes (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL, title TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ingredients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, lemma TEXT, stem TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS recipe_ingredients (recipe_id INTEGER, ingredient_id INTEGER, amount TEXT, full_text TEXT, PRIMARY KEY (recipe_id, ingredient_id), FOREIGN KEY (recipe_id) REFERENCES recipes (id) ON DELETE CASCADE, FOREIGN KEY (ingredient_id) REFERENCES ingredients (id) ON DELETE CASCADE)''')
    conn.commit()
    return conn

def split_ingredient(text):
    text_cleaned = re.sub(r'\(.*?\)', '', text)
    text_cleaned = re.sub(r'[\(\)]+', '', text_cleaned).strip()
    parts = re.split(r'\s+[—–-]\s+', text_cleaned, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text_cleaned, ""

# ==========================================
# 3. ПАРСИНГ ОДНОГО РЕЦЕПТА (только скачивание)
# ==========================================
def parse_povarenok_recipe(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 403: # Если бан
            return {'error': '403'}
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
    except Exception as e:
        return None

# ==========================================
# 4. ПАКЕТНОЕ СОХРАНЕНИЕ В БД
# ==========================================
def save_batch_to_db(conn, recipes_batch):
    if not recipes_batch: return 0
    
    cursor = conn.cursor()
    saved_count = 0
    
    try:
        for recipe_data in recipes_batch:
            # 1. Вставляем рецепт
            cursor.execute("INSERT OR IGNORE INTO recipes (url, title) VALUES (?, ?)", (recipe_data['url'], recipe_data['title']))
            if cursor.rowcount != 1: continue
            
            recipe_id = cursor.lastrowid
            saved_count += 1
            ing_inserts = []
            
            for ing in recipe_data['ingredients']:
                name, amount = split_ingredient(ing)
                if not name: continue
                name = name.replace('(', '').replace(')', '').strip()
                if not name: continue
                
                # Ищем ингредиент
                cursor.execute("SELECT id FROM ingredients WHERE name = ?", (name,))
                row = cursor.fetchone()
                
                if row:
                    ingredient_id = row[0]
                else:
                    lemma, stem = process_ingredient_name(name)
                    cursor.execute("INSERT INTO ingredients (name, lemma, stem) VALUES (?, ?, ?)", (name, lemma, stem))
                    ingredient_id = cursor.lastrowid
                
                ing_inserts.append((recipe_id, ingredient_id, amount, ing))
            
            # Пакетная вставка ингредиентов для рецепта
            if ing_inserts:
                cursor.executemany("INSERT OR IGNORE INTO recipe_ingredients (recipe_id, ingredient_id, amount, full_text) VALUES (?, ?, ?, ?)", ing_inserts)
                
        conn.commit() # Один коммит на всю пачку
        return saved_count
    except sqlite3.Error as e:
        print(f"Ошибка БД при сохранении пачки: {e}")
        conn.rollback()
        return 0

# ==========================================
# 5. УСКОРЕННЫЙ СБОР ССЫЛОК
# ==========================================
def get_links_from_page(page_url):
    """Скачивает одну страницу и вытаскивает ссылки"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        # Уменьшен таймаут до 8 секунд, чтобы не ждать зависшие страницы
        res = requests.get(page_url, headers=headers, timeout=8)
        if res.status_code != 200: 
            return []
            
        soup = BeautifulSoup(res.text, 'html.parser')
        links = set()
        
        # Ищем ссылки на рецепты (используем более быстрый поиск)
        for a in soup.find_all('a', href=True):
            if '/recipes/show/' in a['href']:
                clean_url = urljoin('https://www.povarenok.ru', a['href']).split('#')[0].rstrip('/') + '/'
                links.add(clean_url)
        return list(links)
    except Exception:
        return []

def get_all_site_links():
    """Собирает ссылки со всех страниц пагинации сайта МАКСИМАЛЬНО БЫСТРО"""
    base_url = 'https://www.povarenok.ru/recipes/'
    all_links = set()
    page = 1
    stop_scanning = False
    
    print("Сканируем страницы для сбора ссылок (ускоренный режим)...")
    
    while not stop_scanning:
        # Берем пачку из 20 страниц за один раз (было 10)
        batch_size = 20
        urls_to_scan = [base_url if p == 1 else f"{base_url}~{p}/" for p in range(page, page + batch_size)]
        page += batch_size
        
        found_in_batch = 0
        
        # Увеличиваем количество потоков до 15 (было 10)
        with ThreadPoolExecutor(max_workers=15) as executor: 
            futures = {executor.submit(get_links_from_page, url): url for url in urls_to_scan}
            
            for future in as_completed(futures):
                page_links = future.result()
                # Если страница пустая (404 или конец), ставим флаг остановки
                if not page_links:
                    stop_scanning = True 
                else:
                    for link in page_links:
                        if link not in all_links:
                            all_links.add(link)
                            found_in_batch += 1
                            
        print(f"  Сканировано страниц: {page-1}. Найдено новых ссылок: +{found_in_batch} (всего: {len(all_links)})")
        
        # Если 20 страниц подряд не дали ни одной новой ссылки — значит, мы дошли до конца
        if found_in_batch == 0: 
            stop_scanning = True
            
        # Убираем долгие паузы, оставляем минимальную (0.5 сек), чтобы не забанили
        time.sleep(0.5) 
        
    return list(all_links)

# ==========================================
# 6. ГЛАВНЫЙ ЗАПУСК
# ==========================================
if __name__ == '__main__':
    conn = init_db()
    
    print("Начинаем сбор ВСЕХ ссылок...")
    all_urls = get_all_site_links()
    print(f"\nСБОР ЗАВЕРШЕН. Всего найдено: {len(all_urls)}")
    
    # Фильтруем те, что уже есть в БД
    cursor = conn.cursor()
    new_urls = []
    for url in all_urls:
        cursor.execute("SELECT id FROM recipes WHERE url = ?", (url,))
        if not cursor.fetchone(): new_urls.append(url)
            
    print(f"Новых для скачивания: {len(new_urls)}")
    print("Начинаем МНОГОПОТОЧНЫЙ парсинг рецептов...\n")
    
    total_saved = 0
    batch = []
    
    # Запускаем пул потоков для скачивания рецептов
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Отправляем все задачи на скачивание
        futures = {executor.submit(parse_povarenok_recipe, url): url for url in new_urls}
        
        # По мере завершения скачивания обрабатываем результаты
        for i, future in enumerate(as_completed(futures)):
            data = future.result()
            
            if data and 'error' not in data:
                batch.append(data)
            
            # Когда набралось 50 рецептов, сохраняем их разом
            if len(batch) >= BATCH_SAVE_SIZE:
                print(f"[{i+1}/{len(new_urls)}] Сохраняем пачку из {len(batch)} рецептов...")
                saved = save_batch_to_db(conn, batch)
                total_saved += saved
                print(f"  -> Сохранено в БД: {saved}")
                batch = []
                time.sleep(SLEEP_BETWEEN_BATCHES) # Пауза, чтобы сайт не забанил
                
    # Сохраняем остатки, если они есть
    if batch:
        print(f"Сохраняем последнюю пачку из {len(batch)} рецептов...")
        total_saved += save_batch_to_db(conn, batch)
        
    conn.close()
    print(f"\n=== РАБОТА ЗАВЕРШЕНА. Всего сохранено: {total_saved} ===")
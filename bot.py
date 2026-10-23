import sqlite3
import re
from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger, Doc, Segmenter

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ NLP
# ==========================================
print("Загрузка NLP моделей...")
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# === СЛОВАРЬ СИНОНИМОВ ===
SYNONYMS = {
    'картошка': 'картофель',
    'помидор': 'томат',
    'помидоры': 'томат',
    'томаты': 'томат',
    'куриное': 'курица',
    'куриные': 'курица',
    'яйцо': 'яйца',
    'луковица': 'лук',
}

# === БАЗОВЫЕ ПРОДУКТЫ ===
BASE_INGREDIENTS = {
    'соль', 'вода', 'сахар', 'перец', 'масло', 
    'масло растительное', 'масло сливочное', 
    'специи', 'приправа', 'зелень', 'мука', 'сода', 'уксус'
}

def get_lemmas(text):
    text = re.sub(r'\s+и\s+', ',', text)
    words = [w.strip() for w in text.split(',') if w.strip()]
    
    all_lemmas = []
    for word in words:
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemma = token.lemma if token.lemma else token.text.lower()
            if lemma in SYNONYMS:
                lemma = SYNONYMS[lemma]
            all_lemmas.append(lemma)
            
    return list(set(all_lemmas))

# ==========================================
# 2. ПОИСК РЕЦЕПТОВ (Строгое совпадение)
# ==========================================
def find_recipes_smart(user_ingredients):
    if not user_ingredients:
        return []
        
    conn = sqlite3.connect('recipes.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT r.id, r.title, r.url, i.lemma, i.name
        FROM recipes r
        JOIN recipe_ingredients ri ON r.id = ri.recipe_id
        JOIN ingredients i ON ri.ingredient_id = i.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    recipes_dict = {}
    for row in rows:
        r_id, title, url, lemma, name = row
        if r_id not in recipes_dict:
            recipes_dict[r_id] = {
                'title': title, 
                'url': url, 
                'ingredients': [], 
                'display_ingredients': set() 
            }
        
        # Очищаем лемму от лишних символов
        search_term = lemma.lower().strip() if lemma else name.lower().strip()
        
        # Применяем синонимы
        for syn, base in SYNONYMS.items():
             if syn == search_term:
                 search_term = base
                 
        recipes_dict[r_id]['ingredients'].append((search_term, name))
        recipes_dict[r_id]['display_ingredients'].add(name)
        
    user_set = set(user_ingredients)
    results = []
    
    for r_id, data in recipes_dict.items():
        
        can_cook = True
        user_ingredients_matched = 0 # Счетчик совпадений с пользователем
        
        for ing_lemma, ing_name in data['ingredients']:
            
            # Ищем точное или частичное совпадение с продуктами пользователя
            is_user_has = False
            for u_ing in user_set:
                # Если слово пользователя совпадает с леммой или входит в неё
                if u_ing == ing_lemma or (len(u_ing) > 3 and u_ing in ing_lemma):
                    is_user_has = True
                    user_ingredients_matched += 1
                    break
                    
            if is_user_has:
                continue # Продукт есть у пользователя
            
            # Если нет у пользователя, проверяем БАЗОВЫЕ
            is_base = False
            for b_ing in BASE_INGREDIENTS:
                if b_ing == ing_lemma or (len(b_ing) > 3 and b_ing in ing_lemma):
                    is_base = True
                    break
                    
            if is_base:
                continue # Это базовый продукт, прощаем
                
            # Если продукта нет ни у пользователя, ни в базе — отмена
            can_cook = False
            break
            
        # ГЛАВНОЕ ИСПРАВЛЕНИЕ: 
        # Рецепт подходит, только если он готовится ИЗ ЧЕГО-ТО пользователя.
        # Если совпадений 0 (например, только соль и вода), рецепт не выводим.
        if can_cook and user_ingredients_matched > 0:
            results.append({
                'title': data['title'],
                'url': data['url'],
                'all_ingredients': list(data['display_ingredients'])
            })
            
    results.sort(key=lambda x: len(x['all_ingredients']))
    return results

# ==========================================
# 3. КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# ==========================================
def main():
    print("\n" + "="*50)
    print(" 🍳 ПОВАРЁНОК-БОТ: ЧТО ПРИГОТОВИТЬ?")
    print("="*50)
    
    while True:
        user_input = input("Ваши продукты: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']: break
        if not user_input: continue
            
        user_lemmas = get_lemmas(user_input)
        print(f"🔍 Ищем рецепты из: {', '.join(user_lemmas)}...")
        
        try:
            results = find_recipes_smart(user_lemmas)
        except sqlite3.OperationalError as e:
            print(f"Ошибка БД: {e}")
            break
        
        if not results:
            print("😔 Ничего не нашлось. В базе нет рецептов строго из этих продуктов.\n")
            continue
            
        print(f"\n✅ Найдено подходящих рецептов: {len(results)}\n")
        
        # Показываем топ-5, чтобы не засорять консоль
        for i, r in enumerate(results[:5]):
            print(f"{i+1}. {r['title']}")
            print(f"   Ссылка: {r['url']}")
            ings_str = ", ".join(sorted(r['all_ingredients']))
            print(f"   Ингредиенты: {ings_str}")
            print("-" * 50)
            
        print("\n")

if __name__ == '__main__':
    main()
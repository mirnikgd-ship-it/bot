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

def get_lemmas(text):
    """Превращает запрос пользователя в список лемм (начальных форм)"""
    # Очищаем от лишних символов, разбиваем по запятым или "и"
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
            all_lemmas.append(lemma)
            
    return list(set(all_lemmas)) # Убираем дубликаты

# ==========================================
# 2. УМНЫЙ ПОИСК РЕЦЕПТОВ
# ==========================================
def find_best_recipes(user_ingredients):
    """
    Ищет рецепты. Сначала те, на которые всё есть.
    Если нет — те, на которые нужно докупить минимум.
    """
    if not user_ingredients:
        return []
        
    conn = sqlite3.connect('recipes.db')
    cursor = conn.cursor()
    
    # 1. Получаем ID всех рецептов и их ингредиенты (по леммам)
    # Для оптимизации мы вытаскиваем сразу всю базу в память. 
    # Если рецептов > 100 000, этот шаг может занять 1-2 секунды, но это быстрее, чем 1000 SQL-запросов.
    cursor.execute('''
        SELECT r.id, r.title, r.url, i.lemma 
        FROM recipes r
        JOIN recipe_ingredients ri ON r.id = ri.recipe_id
        JOIN ingredients i ON ri.ingredient_id = i.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    # Группируем ингредиенты по рецептам: {recipe_id: {'title': ..., 'url': ..., 'ingredients': set(...)}}
    recipes_dict = {}
    for row in rows:
        r_id, title, url, lemma = row
        if r_id not in recipes_dict:
            recipes_dict[r_id] = {'title': title, 'url': url, 'ingredients': set()}
        recipes_dict[r_id]['ingredients'].add(lemma)
        
    user_set = set(user_ingredients)
    results = []
    
    # 2. Сравниваем
    for r_id, data in recipes_dict.items():
        recipe_ings = data['ingredients']
        
        # Находим, что у нас есть в рецепте
        matched = user_set.intersection(recipe_ings)
        # Находим, чего не хватает
        missing = recipe_ings - user_set
        
        # Нас интересуют только рецепты, где есть хотя бы 1 наш продукт
        if len(matched) > 0:
            results.append({
                'title': data['title'],
                'url': data['url'],
                'matched': list(matched),
                'missing': list(missing),
                'missing_count': len(missing)
            })
            
    # 3. Сортируем: сначала те, где меньше всего нехватает (missing_count)
    results.sort(key=lambda x: x['missing_count'])
    
    return results

# ==========================================
# 3. КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# ==========================================
def main():
    print("\n" + "="*50)
    print(" 🍳 ПОВАРЁНОК-БОТ: ЧТО ПРИГОТОВИТЬ?")
    print("="*50)
    print("Введите продукты через запятую или 'и'.")
    print("Пример: курица, лук, картошка")
    print("Для выхода напишите 'выход'.\n")
    
    while True:
        user_input = input("Ваши продукты: ").strip()
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("До встречи на кухне!")
            break
            
        if not user_input:
            continue
            
        user_lemmas = get_lemmas(user_input)
        if not user_lemmas:
            print("Не удалось распознать продукты.")
            continue
            
        print(f"🔍 Ищем рецепты с: {', '.join(user_lemmas)}...")
        
        try:
            results = find_best_recipes(user_lemmas)
        except sqlite3.OperationalError as e:
            print(f"Ошибка базы данных: {e}")
            print("Убедитесь, что база создана.")
            break
        
        if not results:
            print("😔 Ничего не нашлось. В базе нет рецептов с этими продуктами.\n")
            continue
            
        # Разделяем результаты на "Идеальные" и "Почти идеальные"
        perfect = [r for r in results if r['missing_count'] == 0]
        
        if perfect:
            print(f"\n✅ У вас есть ВСЁ для {len(perfect)} рецептов! Вот они:")
            print("-" * 50)
            for i, r in enumerate(perfect[:5]): # Показываем топ-5
                print(f"{i+1}. {r['title']}")
                print(f"   Ссылка: {r['url']}")
                print("-" * 50)
        else:
            # Если идеальных нет, показываем те, где нужно докупить минимум (до 3 продуктов)
            print(f"\n⚠️  Для идеальных рецептов не хватает продуктов.")
            print(f"🍳 Вот рецепты, на которые нужно докупить минимум:")
            print("-" * 50)
            
            count = 0
            for r in results:
                # Показываем только те, где не хватает не более 3-х ингредиентов
                if r['missing_count'] <= 3 and count < 5:
                    print(f"{count+1}. {r['title']}")
                    print(f"   ✅ Есть: {', '.join(r['matched'])}")
                    
                    # Переводим леммы нехватки обратно в нормальный вид
                    # (в базе у нас леммы, поэтому выводим их как есть, но с заглавной буквы)
                    missing_str = ", ".join([m.capitalize() for m in r['missing']])
                    print(f"   🛒 Докупить: {missing_str}")
                    print(f"   Ссылка: {r['url']}")
                    print("-" * 50)
                    count += 1
                    
            if count == 0:
                print("Даже с докупкой ничего подходящего не нашлось. Попробуйте добавить другие продукты.\n")

if __name__ == '__main__':
    main()
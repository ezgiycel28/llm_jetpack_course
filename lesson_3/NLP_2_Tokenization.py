import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

try: #bilgisayarda punkt modeli var mı bak yoksa indir demektir
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' paketi indiriliyor...")
    nltk.download('punkt')

text_1 = "kurt kışı geçirir de yediği ayazı unutmaz written by Kurt Baba 2025"
tokens = word_tokenize(text_1) #metni boşluklardan ve noktalama işaretlerinden ayırarak bir liste oluşturur
print(f"Orijinal Metin: {text_1}")
print(f"Tokenlar (Kelimeler): {tokens}\n")

print("--- 2. Sentence Tokenization ---")
text_2 = "After dissecting industry trends, salary reports, and job market data, I’ve identified five AI certifications that actually move the needle in 2025. No fluff. No wasted time. Just your shortcut to dominance."
sentences = sent_tokenize(text_2) #paragrafın içindeki ., !, ? gibi cümle sonu işaretlerini (ve punkt modelindeki kuralları) kullanarak metni cümlelerine ayırır ve bir cümle listesi oluşturur
print(f"Orijinal Metin: {text_2}")
print("Cümleler:")
for i, sent in enumerate(sentences):
    print(f"  {i+1}. {sent}")
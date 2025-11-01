print("--- 1. English Stemming (Snowball) ---")
from nltk.stem.snowball import SnowballStemmer

stemmer_en = SnowballStemmer("english")
words_en = ["running", "runs", "ran", "runner", "easily", "better", "classification"]
stems_en = [stemmer_en.stem(word) for word in words_en]

print(f"Orijinal Kelimeler: {words_en}")
print(f"Kökler (Stems): {stems_en}\n")

print("--- 2. Turkish Stemming (TurkishStemmer) ---")
try:
    from TurkishStemmer import TurkishStemmer
    stemmer_tr = TurkishStemmer()
    
    words_tr = ["okuldakilerden", "gözlükçü", "kitaplarım", "koşuyorum", "yaptılar"]
    stems_tr = [stemmer_tr.stem(word) for word in words_tr]

    print(f"Orijinal Kelimeler: {words_tr}")
    print(f"Kökler (Stems): {stems_tr}")

except ImportError:
    print("HATA: 'TurkishStemmer' kütüphanesi bulunamadı.")
    print("Lütfen terminale 'pip install TurkishStemmer' yazarak yükleyin.")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

text = [ #corpus deniyormuş
    "Uygulama içeriği çok dolu ve güzel hazırlanmış.",
    "Bu kafenin kahveleri çok güzel ve pazar günleri hep doludur"
]

print("--- 1. Bag-of-Words (CountVectorizer) ---")#hangi kelimeden kaç tane var olduğuna odaklanır
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(text)#bir sözlük oluşturur ve bu sözlüğü kullanarak her bir cümleyi bir sayı vektörüne dönüştürür

feature_names_bow = vectorizer_bow.get_feature_names_out()
print(f"Sözlük (Features): {feature_names_bow}")
print("BoW Matrisi:")
print(bow_matrix.toarray())
print("\n")

print("--- 2. TF-IDF (TfidfVectorizer) ---")#BoW'un bir sorunu vardır:her cümlede geçen kelimelere denadir kelimelere de aynı değeri verir. TF-IDF, bu sorunu çözer.
vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(text)

feature_names_tfidf = vectorizer_tfidf.get_feature_names_out()
print(f"Sözlük (Features): {feature_names_tfidf}")
print("TF-IDF Matrisi (Yoğun olmayan formatta):")
print(tfidf_matrix)

# TF (Term Frequency / Terim Sıklığı): Bir kelimenin bir cümlede kaç kez geçtiği.

# IDF (Inverse Document Frequency / Ters Belge Sıklığı): Bir kelimenin tüm cümlelerde ne kadar nadir olduğunu ölçer.
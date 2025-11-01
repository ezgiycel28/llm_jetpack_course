from gensim.models import Word2Vec
from gensim.models import FastText 

sentences = [
    ["bugün", "ders", "çok", "güzel"],
    ["hava", "çok", "güneşli", "ama", "ders", "eğlenceli"],
    ["yarın", "ders", "yok", "ama", "ödev", "var"],
    ["ödev", "çok", "zor", "değil"]
]

print("--- 1. Word2Vec ---")#google tarafından gerçekleştirilmiştir.
model_w2v = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)#window=5: Bir kelimenin anlamını öğrenirken, o kelimenin sağındaki 5 ve solundaki 5 kelimeye bak.
model_w2v.train(sentences, total_examples=len(sentences), epochs=10)                        #min_count=1: Eğer bir kelime metinlerde 1 kez bile geçiyorsa, onu sözlüğe dahil et.
                                                                                            #workers=4: Bu eğitimi yaparken 4 tane işlemci (CPU) çekirdeği kullan.

try:
    similar_to_ders = model_w2v.wv.most_similar("ders")
    print(f"'ders' kelimesine en benzer kelimeler:\n{similar_to_ders}\n")
except KeyError:
    print("'ders' kelimesi sözlükte bulunamadı.\n")


print("--- 2. FastText ---")#facebook tarafından geliştirilmiştir.
model_ft = FastText(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4) #window=5: Bir kelimenin anlamını öğrenirken, o kelimenin sağındaki 5 ve solundaki 5 kelimeye bak.
model_ft.train(sentences, total_examples=len(sentences), epochs=10)                         #min_count=1: Eğer bir kelime metinlerde 1 kez bile geçiyorsa, onu sözlüğe dahil et.
                                                                                            #workers=4: Bu eğitimi yaparken 4 tane işlemci (CPU) çekirdeği kullan.

try:
    similar_to_odev = model_ft.wv.most_similar("ödev")
    print(f"'ödev' kelimesine en benzer kelimeler:\n{similar_to_odev}\n")
except KeyError:
    print("'ödev' kelimesi sözlükte bulunamadı.\n")

try:
    similar_to_typo = model_ft.wv.most_similar("güneşli")
    print(f"Sözlükte olmayan 'güneşli' kelimesine benzerler:\n{similar_to_typo}")
except KeyError:
    print("Model 'güneşliii' için bile benzerlik bulamadı.")
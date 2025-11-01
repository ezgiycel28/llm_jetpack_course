import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'punkt' ve 'stopwords' paketleri indiriliyor...")
    nltk.download('punkt')
    nltk.download('stopwords') 

stop_words = stopwords.words("turkish")#türkçe stopwords kelimelerini yükler

print("--- Örnek 1 ---")
text_1 = "Turkish Student Co bir öğrenci topluluğudur ve öğrencilerden oluşur."
tokens_1 = word_tokenize(text_1)
clean_tokens_1 = [word for word in tokens_1 if word.lower() not in stop_words]#kelimemizin küçük harfli hali stop_words kelimelerimiin içinde yoksa ekle

print(f"Orijinal Tokenlar: {tokens_1}")
print(f"Temizlenmiş Tokenlar: {clean_tokens_1}\n")


print("--- Örnek 2 ---")
text_2 = "Bugün derse gittim ama orada çok fazla birşey öğrenmedim ve sonra sıkılıp amcam ile birlikte eve döndük."
tokens_2 = word_tokenize(text_2)
clean_tokens_2 = [word for word in tokens_2 if word.lower() not in stop_words]#aynı mantıktır

print(f"Orijinal Tokenlar: {tokens_2}")
print(f"Temizlenmiş Tokenlar: {clean_tokens_2}")
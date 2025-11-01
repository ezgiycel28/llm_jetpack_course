import string
import re

text_1 = "Ezgi YÜCEL iyi BİR araştırmacıdır!"
text_1_lower = text_1.lower()
print(f"Orijinal: {text_1}")
print(f"Küçültülmüş: {text_1_lower}\n")

text_2 = "Merhaba Mehmet, Proje ne durumda ?"
translator = str.maketrans('', '', string.punctuation)
clean_text_2 = text_2.translate(translator)
print(f"Orijinal: {text_2}")
print(f"Temizlenmiş: {clean_text_2}\n")

text_3 = "kurt kışı geçirir de yediği ayazı unutmaz @written by Kurt Baba #2025"
clean_text_3 = re.sub(r'(@[^\s]+)|(#[^\s]+)|(http\S+)|(\d+)', '', text_3)
print(f"Orijinal: {text_3}")
print(f"Temizlenmiş: {clean_text_3}\n")

text_4 = "Kurt   bu   nasıl   tweeet"
clean_text_4 = " ".join(text_4.split()) 
print(f"Orijinal: {text_4}")
print(f"Temizlenmiş: {clean_text_4}\n")
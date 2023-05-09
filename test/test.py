from langdetect import detect as detect_language

print(detect_language("你好。"))
print(detect_language("english,"))
print(detect_language("english,中国"))
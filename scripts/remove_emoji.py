import os

file_path = 'c:/Users/Sueb/Desktop/NLP_Local/dashboard.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    'page_icon="🚀",': 'page_icon="bar_chart",',
    '"🌟 NLP Sentiment Analysis Dashboard"': '"NLP Sentiment Analysis Dashboard"',
    '🏆 Best Model': 'Best Model',
    '🔥 Best F1-Macro': 'Best F1-Macro',
    '🎯 Accuracy': 'Accuracy',
    '📊 Total Models': 'Total Models',
    '📈 F1-Macro Comparison': 'F1-Macro Comparison',
    '📋 Detail Peringkat Model': 'Detail Peringkat Model',
    '🧩 Confusion Matrix': 'Confusion Matrix',
    '❤️ menggunakan': 'menggunakan'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Emojis removed successfully.')

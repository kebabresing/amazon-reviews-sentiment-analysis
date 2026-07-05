import os

# 1. Update dashboard.py paths
with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    "'all_metrics.json'": "'results/all_metrics.json'",
    "'distilbert_history.png'": "'visualizations/distilbert_history.png'",
    "'glove300d_results.png'": "'visualizations/glove300d_results.png'",
    "'distilbert_cm.png'": "'visualizations/distilbert_cm.png'",
    "'comparison_confusion_matrices.png'": "'visualizations/comparison_confusion_matrices.png'",
    "'comparison_all_metrics.png'": "'visualizations/comparison_all_metrics.png'",
    "'comparison_radar.png'": "'visualizations/comparison_radar.png'",
    "'comparison_f1_ranking.png'": "'visualizations/comparison_f1_ranking.png'",
    "'final_comparison.png'": "'visualizations/final_comparison.png'",
    "'vectorizer_ngram.pkl'": "'models/vectorizer_ngram.pkl'",
    "'model_nb_ngram.pkl'": "'models/model_nb_ngram.pkl'",
}

for k, v in replacements.items():
    content = content.replace(k, v)

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 2. Update .gitignore
with open('.gitignore', 'a', encoding='utf-8') as f:
    f.write("\n# Models for Streamlit Deployment\n!models/model_nb_ngram.pkl\n!models/vectorizer_ngram.pkl\n")

# 3. Update README.md
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

tree_start = readme.find("nlp-sentiment-amazon/\n│")
if tree_start != -1:
    tree_end = readme.find("```", tree_start)
    new_tree = """nlp-sentiment-amazon/
│
├── 🎯 dashboard.py                   # Streamlit interactive dashboard
├── 📋 requirements.txt
├── .gitignore
├── README.md
│
├── 📓 notebooks/
│   └── AIO_Pipeline.ipynb            # Pipeline NLP lengkap
│
├── 📊 results/
│   ├── all_metrics.json              # Hasil evaluasi semua model
│   └── *_metrics.json
│
├── 🖼️ visualizations/
│   └── *.png                         # Grafik comparison & confusion matrix
│
├── 🤖 models/
│   ├── model_nb_ngram.pkl            # Model yang di-deploy (Inference)
│   ├── vectorizer_ngram.pkl
│   └── (model & tokenizer lainnya)
│
├── 📁 data/
│   └── (Dataset CSV & GloVe - diabaikan Git)
│
└── 🔧 scripts/
    └── *.py                          # Script utility & preprocessing
"""
    readme = readme[:tree_start] + new_tree + readme[tree_end:]
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)

print("Files updated successfully")

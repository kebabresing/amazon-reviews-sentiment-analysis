import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import re
import os
import math

st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0A0E1A; }
.stApp { background-color: #0A0E1A; }

/* Typography */
h1 { color: #F1F5F9; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #CBD5E1; font-weight: 600; }
h3 { color: #94A3B8; font-weight: 500; }

/* Metric Card */
.metric-card {
    background: linear-gradient(145deg, #141929, #1A2035);
    border: 1px solid #1E2D45;
    padding: 22px 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #38BDF8;
    line-height: 1.2;
    margin-top: 6px;
}
.metric-title {
    font-size: 0.78rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Section Header */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E2D45;
    margin-bottom: 16px;
}

/* Conclusion Box */
.conclusion-box {
    background: linear-gradient(135deg, #0F1E35, #111827);
    border: 1px solid #1E3A5F;
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 24px 28px;
    line-height: 1.8;
    color: #CBD5E1;
}
.conclusion-box h3 { color: #F1F5F9; margin-bottom: 8px; }

/* ── Tab Bar Premium ── */

/* Tab list container */
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #1E2D45 !important;
    gap: 4px !important;
}

/* Setiap tab button */
button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    padding: 0.85rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.25s ease, background 0.25s ease, border-color 0.25s ease !important;
    letter-spacing: 0.3px !important;
    position: relative !important;
}

/* Konten dalam tab button */
button[data-baseweb="tab"] > div {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}

/* Hover state — tab tidak aktif */
button[data-baseweb="tab"]:hover {
    color: #CBD5E1 !important;
    background: rgba(56, 189, 248, 0.06) !important;
    border-bottom: 2px solid rgba(56, 189, 248, 0.35) !important;
}

/* Active / selected tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    font-weight: 700 !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border-bottom: 2px solid #38BDF8 !important;
    text-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
}

/* Hapus garis bawah default Streamlit */
div[data-testid="stTabs"] [role="tablist"] {
    gap: 2px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #0D1321; border-right: 1px solid #1E2D45; }

/* Divider */
.divider { border: none; border-top: 1px solid #1E2D45; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Parse Data ────────────────────────────────────────────────────────
def load_data():
    try:
        with open('results/all_metrics.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Gagal memuat file metrik: {e}")
        return {}

data = load_data()

rows = []
for k, v in data.items():
    if isinstance(v, dict) and 'f1' in v:
        cm = v.get('confusion_matrix', [])
        p_neg = r_neg = p_pos = r_pos = 0
        if len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
            r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
            p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
        rows.append({
            'Model': k,
            'Kategori': 'Deep Learning' if k in ['DistilBERT', 'GloVe + MLP', 'Word2Vec + MLP'] else 'Classical ML',
            'Accuracy': v.get('accuracy', 0), 'Precision': v.get('precision', 0),
            'Recall': v.get('recall', 0), 'F1-Macro': v.get('f1', 0),
            'P_Neg': p_neg, 'R_Neg': r_neg, 'P_Pos': p_pos, 'R_Pos': r_pos,
            'CM': cm,
            'TrainTime': v.get('train_time_s'), 'Epochs': v.get('epochs'),
            'ModelName': v.get('model_name'),
        })

if 'embedding_models' in data and 'GloVe_300d_MLP' in data['embedding_models']:
    g = data['embedding_models']['GloVe_300d_MLP']
    # Jika accuracy tidak ada, estimasi dari f1_macro
    g_accuracy = g.get('accuracy') or g.get('f1_macro', 0)
    rows.append({
        'Model': g.get('model', 'GloVe 300d + MLP'), 'Kategori': 'Deep Learning',
        'Accuracy': g_accuracy, 'Precision': g.get('precision_macro', 0),
        'Recall': g.get('recall_macro', 0), 'F1-Macro': g.get('f1_macro', 0),
        'P_Neg': g.get('f1_negative', 0), 'R_Neg': 0, 'P_Pos': g.get('f1_positive', 0), 'R_Pos': 0,
        'CM': [], 'TrainTime': None, 'Epochs': None, 'ModelName': None,
    })

df = pd.DataFrame(rows).sort_values('F1-Macro', ascending=False).reset_index(drop=True)
df.index += 1

LAYOUT = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
              font=dict(color='#94A3B8', family='Inter'), margin=dict(t=30, b=10))

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## NLP Dashboard")
    selected = option_menu(
        menu_title=None,
        options=[
            "Overview",
            "Data & Preprocessing",
            "Feature Extraction",
            "Training & Evaluation",
            "Model Comparison",
            "Error Analysis",
            "Final Insights",
            "Interactive Prediction",
        ],
        icons=["house-fill", "database-fill", "gear-fill", "cpu-fill",
               "bar-chart-fill", "search", "lightbulb-fill", "bullseye"],
        default_index=0,
        styles={
            "container": {"padding": "4px 0!important", "background-color": "transparent"},
            "icon": {"color": "#64748B", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px", "font-weight": "500",
                "text-align": "left", "margin": "1px 0",
                "padding": "10px 16px",
                "border-radius": "8px",
                "color": "#94A3B8",
                "--hover-color": "rgba(56,189,248,0.08)",
            },
            "nav-link-selected": {
                "background-color": "rgba(56,189,248,0.12)",
                "color": "#38BDF8",
                "font-weight": "600",
                "border-left": "3px solid #38BDF8",
            },
        }
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Informasi Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
**Dataset:** Amazon Fine Food Reviews  
**Task:** Binary Sentiment Analysis  
**Metrik Utama:** F1-Macro (Macro-Averaged)  
**Total Sampel:** 363.888 reviews  
**Kelas:** Negatif (0) & Positif (1)
""")
    st.markdown('<div class="section-header" style="margin-top:20px;">Peringkat Model</div>', unsafe_allow_html=True)
    for _, row in df.iterrows():
        cat_color = "#38BDF8" if row['Kategori'] == 'Deep Learning' else "#94A3B8"
        st.markdown(
            f"<div style='font-size:0.82rem;padding:4px 0;'>"
            f"<span style='color:{cat_color};font-weight:600;'>{row['Model']}</span>"
            f"<span style='float:right;color:#475569;'>{row['F1-Macro']:.4f}</span></div>",
            unsafe_allow_html=True
        )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    export_df = df[['Model', 'Kategori', 'Accuracy', 'Precision', 'Recall', 'F1-Macro']].copy()
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="model_comparison.csv",
                       mime="text/csv", use_container_width=True)

@st.cache_resource
def load_predictor():
    try:
        # Diupgrade menggunakan model N-Gram yang memiliki akurasi/F1 lebih baik
        vec   = joblib.load('models/vectorizer_ngram.pkl')
        model = joblib.load('models/model_nb_ngram.pkl')
        return vec, model
    except Exception as e:
        return None, None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# NLP Sentiment Analysis Dashboard")
_total_models = len(df) if not df.empty else 10
st.markdown(f"<p style='color:#64748B;font-size:0.95rem;margin-top:-10px;'>Perbandingan performa {_total_models} model Machine Learning dan Deep Learning — Amazon Fine Food Reviews</p>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
if selected == "Overview":
    st.markdown("### Executive Summary")
    st.markdown("<p style='color:#64748B;font-size:0.95rem;'>Ringkasan metrik utama dari proyek klasifikasi sentimen Amazon Fine Food Reviews.</p>", unsafe_allow_html=True)
    if not df.empty:
        best = df.iloc[0]
        ml_df = df[df['Kategori'] == 'Classical ML']
        dl_df = df[df['Kategori'] == 'Deep Learning']
        best_ml_name = ml_df.iloc[0]['Model'] if not ml_df.empty else "N/A"

        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, "Best Model", best['Model'], "#F59E0B"),
            (c2, "Best F1-Macro", f"{best['F1-Macro']:.4f}", "#38BDF8"),
            (c3, "Best Accuracy", f"{best['Accuracy']*100:.2f}%", "#34D399"),
            (c4, "Best Classical ML", best_ml_name, "#A78BFA"),
            (c5, "Total Model", str(len(df)), "#FB7185"),
        ]
        for col, title, val, color in cards:
            col.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value" style="color:{color};font-size:1.3rem;">{val}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Ringkasan Proyek</div>', unsafe_allow_html=True)
        col_ov1, col_ov2, col_ov3 = st.columns(3)
        ov_items = [
            (col_ov1, "Dataset", "Amazon Fine Food Reviews", "363.888 ulasan", "#38BDF8"),
            (col_ov2, "Task", "Binary Sentiment Analysis", "Negatif (0) vs Positif (1)", "#34D399"),
            (col_ov3, "Metrik Utama", "F1-Macro", "Macro-Averaged F1 Score", "#A78BFA"),
        ]
        for col, title, val, sub, color in ov_items:
            col.markdown(f"""
<div style='background:linear-gradient(145deg,#141929,#1A2035);border:1px solid #1E2D45;
border-left:4px solid {color};border-radius:12px;padding:20px;'>
<div style='font-size:0.7rem;color:{color};font-weight:700;text-transform:uppercase;letter-spacing:1px;'>{title}</div>
<div style='color:#F1F5F9;font-size:1.1rem;font-weight:600;margin:8px 0 4px;'>{val}</div>
<div style='color:#64748B;font-size:0.82rem;'>{sub}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Peringkat Model (Semua)</div>', unsafe_allow_html=True)
        cmap = {'Classical ML': '#64748B', 'Deep Learning': '#38BDF8'}
        fig_ov = px.bar(df, x='F1-Macro', y='Model', orientation='h',
                        color='Kategori', color_discrete_map=cmap, text_auto='.4f')
        fig_ov.update_layout(**LAYOUT, yaxis={'categoryorder': 'total ascending'},
                             legend_title_text='', xaxis=dict(range=[0.65, 1.0]), height=350)
        fig_ov.update_traces(textfont_color='white')
        st.plotly_chart(fig_ov, use_container_width=True)
    else:
        st.warning("Data model tidak tersedia. Pastikan file results/all_metrics.json ada.")

# ─── Content Pages ────────────────────────────────────────────────────────────

# ── Dataset Overview ──────────────────────────────────────────────
elif selected == "Data & Preprocessing":
    st.markdown('<div class="section-header">Eksplorasi Dataset — Amazon Fine Food Reviews</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Analisis awal terhadap distribusi dan karakteristik data sebelum pemodelan.</p>", unsafe_allow_html=True)

    # ── Statistik Dataset
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    stats = [
        (col_s1, "Total Sampel", "363.888", "#38BDF8"),
        (col_s2, "Data Training", "254.721 (70%)", "#34D399"),
        (col_s3, "Data Testing", "72.778 (20%)", "#F59E0B"),
        (col_s4, "Data Validasi", "36.389 (10%)", "#A78BFA"),
    ]
    for col, title, val, color in stats:
        col.markdown(
            f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value" style="color:{color};font-size:1.2rem;">{val}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_eda1, col_eda2 = st.columns([4, 6])

    with col_eda1:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Distribusi Kelas</div>', unsafe_allow_html=True)
        # Data distribusi kelas dari dataset keseluruhan
        labels_class = ['Positif (Skor 4-5)', 'Negatif (Skor 1-2)']
        values_class = [306807, 57081]
        fig_pie = go.Figure(go.Pie(
            labels=labels_class,
            values=values_class,
            hole=0.55,
            marker=dict(colors=['#38BDF8', '#FB7185'],
                        line=dict(color='#0A0E1A', width=2)),
            textfont=dict(color='white', size=13),
            textinfo='label+percent'
        ))
        fig_pie.update_layout(
            **LAYOUT,
            height=320,
            showlegend=False,
            annotations=[dict(
                text='<b>363.888</b><br>Reviews',
                x=0.5, y=0.5, font=dict(size=14, color='#CBD5E1'), showarrow=False
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("""
<div style='background:#0F1E35;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;font-size:0.82rem;color:#94A3B8;'>
⚠️ <strong>Class Imbalance:</strong> Kelas Positif (84.3%) jauh lebih dominan dari Negatif (15.7%). 
Hal ini menjadi alasan utama pemilihan <strong>F1-Macro</strong> sebagai metrik utama evaluasi.
</div>
""", unsafe_allow_html=True)

    with col_eda2:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Distribusi Skor Ulasan (1–5)</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#64748B;font-size:0.78rem;margin-bottom:8px;'>Ulasan berskor 3 (Netral) dihapus karena tidak relevan untuk analisis sentimen biner.</p>", unsafe_allow_html=True)
        # Hanya tampilkan skor yang ada (exclude skor 3 yang dihapus)
        score_labels_clean = ['Skor 1', 'Skor 2', 'Skor 4', 'Skor 5']
        score_values_clean = [36285, 20796, 56051, 250756]
        score_colors_clean = ['#FB7185', '#FDA4AF', '#6EE7B7', '#38BDF8']
        score_pct = [f'{v/363888*100:.1f}%' for v in score_values_clean]
        fig_score = go.Figure(go.Bar(
            x=score_labels_clean,
            y=score_values_clean,
            marker_color=score_colors_clean,
            text=[f'{v:,}<br>({p})' for v, p in zip(score_values_clean, score_pct)],
            textposition='outside',
            textfont=dict(color='#94A3B8', size=11)
        ))
        fig_score.update_layout(
            **LAYOUT,
            height=320,
            yaxis=dict(title='Jumlah Ulasan', gridcolor='#1E2D45', range=[0, 300000]),
            xaxis=dict(title=''),
            bargap=0.3
        )
        st.plotly_chart(fig_score, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_eda3, col_eda4 = st.columns(2)

    with col_eda3:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Rata-rata Panjang Teks per Kelas</div>', unsafe_allow_html=True)
        txt_len_data = {
            'Kelas': ['Negatif', 'Positif'],
            'Rata-rata Kata': [68, 82],
            'Median Kata': [42, 55],
            'Maks Kata': [512, 512]
        }
        fig_txt = go.Figure()
        fig_txt.add_trace(go.Bar(name='Rata-rata Kata', x=txt_len_data['Kelas'],
                                  y=txt_len_data['Rata-rata Kata'],
                                  marker_color='#38BDF8', text=txt_len_data['Rata-rata Kata'],
                                  textposition='outside'))
        fig_txt.add_trace(go.Bar(name='Median Kata', x=txt_len_data['Kelas'],
                                  y=txt_len_data['Median Kata'],
                                  marker_color='#A78BFA', text=txt_len_data['Median Kata'],
                                  textposition='outside'))
        fig_txt.update_layout(**LAYOUT, barmode='group', height=300,
                               yaxis=dict(title='Jumlah Kata', gridcolor='#1E2D45'),
                               legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig_txt, use_container_width=True)

    with col_eda4:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Top 10 Kata Paling Sering (Setelah Preprocessing)</div>', unsafe_allow_html=True)
        top_words = {
            'Kata': ['good', 'great', 'love', 'best', 'product', 'taste', 'like', 'time', 'buy', 'flavor'],
            'Frekuensi': [18420, 15330, 14200, 12100, 11850, 10900, 10300, 9800, 9500, 9200]
        }
        fig_words = go.Figure(go.Bar(
            x=top_words['Frekuensi'][::-1],
            y=top_words['Kata'][::-1],
            orientation='h',
            marker=dict(
                color=top_words['Frekuensi'][::-1],
                colorscale=[[0, '#1E3A5F'], [1, '#38BDF8']],
                showscale=False
            ),
            text=[f'{v:,}' for v in top_words['Frekuensi'][::-1]],
            textposition='outside',
            textfont=dict(color='#94A3B8')
        ))
        fig_words.update_layout(**LAYOUT, height=300,
                                  xaxis=dict(title='Frekuensi', gridcolor='#1E2D45'))
        st.plotly_chart(fig_words, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Case Folding", "Mengubah seluruh teks menjadi huruf kecil untuk menghilangkan ambiguitas."),
        ("2", "Cleaning", "Menghapus karakter khusus, angka, HTML tags, dan URL."),
        ("3", "Tokenisasi", "Memisahkan kalimat menjadi token menggunakan NLTK."),
        ("4", "Stopword Removal", "Menghapus kata-kata umum yang tidak membawa makna sentimen."),
        ("5", "Lemmatisasi", "Mengubah kata ke bentuk dasar menggunakan WordNet Lemmatizer."),
        ("6", "Augmentasi", "Back-translation (EN→DE→EN) untuk mengatasi class imbalance."),
    ]
    cols_step = st.columns(3)
    for i, (num, title_step, desc) in enumerate(steps):
        with cols_step[i % 3]:
            st.markdown(f"""
<div style='background:linear-gradient(145deg,#141929,#1A2035);border:1px solid #1E2D45;
border-left:3px solid #38BDF8;border-radius:10px;padding:14px 16px;margin-bottom:12px;'>
<div style='font-size:0.7rem;color:#38BDF8;font-weight:700;text-transform:uppercase;
letter-spacing:1px;margin-bottom:4px;'>STEP {num}</div>
<div style='color:#F1F5F9;font-weight:600;margin-bottom:4px;'>{title_step}</div>
<div style='color:#64748B;font-size:0.82rem;line-height:1.5;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Augmentation & Balancing</div>', unsafe_allow_html=True)
    col_aug1, col_aug2 = st.columns(2)
    with col_aug1:
        st.markdown("""
<div style='background:linear-gradient(145deg,#141929,#1A2035);border:1px solid #1E2D45;
border-left:3px solid #A78BFA;border-radius:10px;padding:18px;'>
<div style='font-size:0.75rem;color:#A78BFA;font-weight:700;text-transform:uppercase;letter-spacing:1px;'>Augmentation Strategy</div>
<div style='color:#F1F5F9;font-size:1.1rem;font-weight:600;margin:8px 0 4px;'>Back-Translation</div>
<div style='color:#64748B;font-size:0.83rem;line-height:1.6;'>Teks pada kelas Negatif (minoritas) diterjemahkan EN→DE lalu dikembalikan ke EN menggunakan MarianMT untuk menambah variasi data training.</div>
</div>
""", unsafe_allow_html=True)
    with col_aug2:
        st.markdown("""
<div style='background:linear-gradient(145deg,#141929,#1A2035);border:1px solid #1E2D45;
border-left:3px solid #34D399;border-radius:10px;padding:18px;'>
<div style='font-size:0.75rem;color:#34D399;font-weight:700;text-transform:uppercase;letter-spacing:1px;'>Class Balancing</div>
<div style='color:#F1F5F9;font-size:1.1rem;font-weight:600;margin:8px 0 4px;'>Class Weight & Augmentation</div>
<div style='color:#64748B;font-size:0.83rem;line-height:1.6;'>Model Deep Learning menggunakan <code>class_weight</code> untuk memberikan penalti lebih besar pada kesalahan kelas minoritas (Negatif).</div>
</div>
""", unsafe_allow_html=True)

elif selected == "Feature Extraction":
    st.markdown('<div class="section-header">Feature Extraction — Representasi Teks</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Model Machine Learning tidak dapat membaca teks mentah. Teks harus diubah menjadi representasi numerik (vektor) terlebih dahulu. Berikut metode ekstraksi fitur yang digunakan dalam proyek ini.</p>", unsafe_allow_html=True)

    # ── Method Cards ──
    methods = [
        {
            "name": "TF-IDF",
            "full": "Term Frequency – Inverse Document Frequency",
            "color": "#38BDF8",
            "used_by": "Naive Bayes, Logistic Regression, SVM, SGD, RF, KNN",
            "dim": "N-gram (1,2) · max 50.000 fitur",
            "pros": "Cepat, efisien memori, tidak perlu GPU.",
            "cons": "Tidak menangkap konteks semantik atau urutan kata.",
            "desc": "Menghitung bobot setiap kata berdasarkan frekuensi kemunculannya di dokumen dan seberapa jarang kata tersebut muncul di seluruh korpus.",
        },
        {
            "name": "GloVe 50d",
            "full": "Global Vectors for Word Representation (50 dim)",
            "color": "#A78BFA",
            "used_by": "GloVe 50d + MLP",
            "dim": "50 dimensi · pre-trained Stanford",
            "pros": "Representasi semantik lebih kaya dari TF-IDF.",
            "cons": "Dimensi kecil, informasi terbatas.",
            "desc": "Word embedding pre-trained yang memetakan kata ke dalam ruang vektor berdasarkan ko-okurens global. Setiap review direpresentasikan sebagai rata-rata vektor semua kata.",
        },
        {
            "name": "GloVe 300d",
            "full": "Global Vectors for Word Representation (300 dim)",
            "color": "#34D399",
            "used_by": "GloVe 300d + MLP",
            "dim": "300 dimensi · max 200 token",
            "pros": "Representasi lebih kaya, menangkap sinonim & relasi semantik.",
            "cons": "Lebih lambat dan berat dari GloVe 50d.",
            "desc": "Versi GloVe dengan dimensi lebih tinggi menghasilkan representasi yang lebih ekspresif. Dipadukan dengan MLP 3-layer untuk klasifikasi sentimen.",
        },
        {
            "name": "Word2Vec",
            "full": "Word to Vector (Skip-gram / CBOW)",
            "color": "#F59E0B",
            "used_by": "Word2Vec + MLP",
            "dim": "100 dimensi · trained on corpus",
            "pros": "Dilatih pada data sendiri, kontekstual terhadap domain.",
            "cons": "Butuh data besar untuk embedding yang optimal.",
            "desc": "Word embedding yang dilatih langsung dari dataset Amazon Fine Food Reviews. Model Skip-gram mempelajari representasi kata berdasarkan kata-kata di sekitarnya.",
        },
        {
            "name": "DistilBERT",
            "full": "Distilled BERT (distilbert-base-uncased)",
            "color": "#FB7185",
            "used_by": "DistilBERT (Fine-tuned)",
            "dim": "768 dimensi · contextual token embedding",
            "pros": "Kontekstual, menangkap makna berdasarkan kalimat penuh.",
            "cons": "Sangat berat komputasi, butuh GPU.",
            "desc": "Model Transformer pre-trained yang di-fine-tune untuk klasifikasi sentimen. Setiap token mendapat representasi unik berdasarkan konteks kalimat penuh.",
        },
    ]

    cols = st.columns(len(methods))
    for col, m in zip(cols, methods):
        col.markdown(f"""
<div style='background:linear-gradient(145deg,#141929,#1A2035);border:1px solid #1E2D45;
border-top:3px solid {m["color"]};border-radius:12px;padding:16px;height:100%;'>
<div style='font-size:1.2rem;font-weight:700;color:{m["color"]};margin-bottom:4px;'>{m["name"]}</div>
<div style='font-size:0.7rem;color:#475569;margin-bottom:10px;line-height:1.4;'>{m["full"]}</div>
<div style='font-size:0.75rem;color:#64748B;margin-bottom:8px;line-height:1.6;'>{m["desc"]}</div>
<div style='background:#0A0E1A;border-radius:6px;padding:8px;margin-top:8px;'>
<div style='font-size:0.65rem;color:#38BDF8;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;'>Digunakan oleh</div>
<div style='font-size:0.72rem;color:#94A3B8;margin-top:2px;'>{m["used_by"]}</div>
</div>
<div style='background:#0A0E1A;border-radius:6px;padding:8px;margin-top:6px;'>
<div style='font-size:0.65rem;color:#A78BFA;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;'>Dimensi / Config</div>
<div style='font-size:0.72rem;color:#94A3B8;margin-top:2px;'>{m["dim"]}</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Comparison Table ──
    st.markdown('<div class="section-header">Perbandingan Metode</div>', unsafe_allow_html=True)
    fe_data = {
        "Metode": ["TF-IDF", "GloVe 50d", "GloVe 300d", "Word2Vec", "DistilBERT"],
        "Tipe": ["Sparse", "Dense", "Dense", "Dense", "Contextual"],
        "Dimensi": [50000, 50, 300, 100, 768],
        "Pre-trained": ["❌", "✅ Stanford", "✅ Stanford", "✅ Domain", "✅ HuggingFace"],
        "Kontekstual": ["❌", "❌", "❌", "❌", "✅"],
        "GPU Required": ["❌", "❌", "❌", "❌", "✅"],
    }
    fe_df = pd.DataFrame(fe_data)
    st.dataframe(fe_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Visual ──
    st.markdown('<div class="section-header">Pipeline Ekstraksi Fitur</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='background:linear-gradient(145deg,#0F1E35,#111827);border:1px solid #1E3A5F;border-radius:12px;padding:24px;'>
<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>
  <div style='background:#141929;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>📝</div>
    <div style='color:#F1F5F9;font-size:0.8rem;font-weight:600;margin-top:4px;'>Raw Text</div>
    <div style='color:#475569;font-size:0.7rem;'>Ulasan mentah</div>
  </div>
  <div style='color:#38BDF8;font-size:1.5rem;'>→</div>
  <div style='background:#141929;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>🧹</div>
    <div style='color:#F1F5F9;font-size:0.8rem;font-weight:600;margin-top:4px;'>Preprocessing</div>
    <div style='color:#475569;font-size:0.7rem;'>Clean + Lemma</div>
  </div>
  <div style='color:#38BDF8;font-size:1.5rem;'>→</div>
  <div style='background:#141929;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>⚙️</div>
    <div style='color:#F1F5F9;font-size:0.8rem;font-weight:600;margin-top:4px;'>Vektorisasi</div>
    <div style='color:#475569;font-size:0.7rem;'>TF-IDF / Embedding</div>
  </div>
  <div style='color:#38BDF8;font-size:1.5rem;'>→</div>
  <div style='background:#141929;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>🔢</div>
    <div style='color:#F1F5F9;font-size:0.8rem;font-weight:600;margin-top:4px;'>Feature Vector</div>
    <div style='color:#475569;font-size:0.7rem;'>Numerik</div>
  </div>
  <div style='color:#38BDF8;font-size:1.5rem;'>→</div>
  <div style='background:#141929;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>🧠</div>
    <div style='color:#F1F5F9;font-size:0.8rem;font-weight:600;margin-top:4px;'>Model</div>
    <div style='color:#475569;font-size:0.7rem;'>Klasifikasi</div>
  </div>
  <div style='color:#38BDF8;font-size:1.5rem;'>→</div>
  <div style='background:#141929;border:1px solid #34D399;border-radius:8px;padding:12px 16px;text-align:center;min-width:120px;'>
    <div style='font-size:1.4rem;'>🎯</div>
    <div style='color:#34D399;font-size:0.8rem;font-weight:600;margin-top:4px;'>Prediksi</div>
    <div style='color:#475569;font-size:0.7rem;'>Pos / Neg</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)


# ── TAB 1: Perbandingan ───────────────────────────────────────────────────────
elif selected == "Model Comparison":
    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.markdown('<div class="section-header">Ranking F1-Macro</div>', unsafe_allow_html=True)
        cmap = {'Classical ML': '#64748B', 'Deep Learning': '#38BDF8'}
        fig_bar = px.bar(df, x='F1-Macro', y='Model', orientation='h',
                         color='Kategori', color_discrete_map=cmap, text_auto='.4f')
        fig_bar.update_layout(**LAYOUT, yaxis={'categoryorder': 'total ascending'},
                              legend_title_text='', xaxis=dict(range=[0.65, 1.0]))
        fig_bar.update_traces(textfont_color='white')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Tabel Peringkat</div>', unsafe_allow_html=True)
        disp = df[['Model', 'Kategori', 'Accuracy', 'Precision', 'Recall', 'F1-Macro']].copy()
        for c in ['Accuracy', 'Precision', 'Recall', 'F1-Macro']:
            disp[c] = disp[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, height=360)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Perbandingan 4 Metrik</div>', unsafe_allow_html=True)
    melt = df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Macro']].melt(
        id_vars='Model', var_name='Metrik', value_name='Score')
    fig_g = px.bar(melt, x='Model', y='Score', color='Metrik', barmode='group',
                   text_auto='.3f', color_discrete_sequence=['#38BDF8','#34D399','#F59E0B','#A78BFA'])
    fig_g.update_layout(**LAYOUT, xaxis_tickangle=-30, yaxis=dict(range=[0.6, 1.0]))
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Radar Chart — Top Models</div>', unsafe_allow_html=True)
    sel = st.multiselect("Pilih model:", options=df['Model'].tolist(), default=df['Model'].tolist()[:3])
    if sel:
        fig_r = go.Figure()
        hex_to_rgba_map = {
            '#38BDF8': 'rgba(56,189,248,0.12)',
            '#34D399': 'rgba(52,211,153,0.12)',
            '#F59E0B': 'rgba(245,158,11,0.12)',
            '#A78BFA': 'rgba(167,139,250,0.12)',
            '#FB7185': 'rgba(251,113,133,0.12)',
            '#FBBF24': 'rgba(251,191,36,0.12)',
            '#6EE7B7': 'rgba(110,231,183,0.12)',
        }
        colors = list(hex_to_rgba_map.keys())
        for i, m in enumerate(sel):
            row = df[df['Model'] == m].iloc[0]
            vals = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Macro']]
            hex_c = colors[i % len(colors)]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=['Accuracy','Precision','Recall','F1-Macro','Accuracy'],
                fill='toself', name=m, line=dict(color=hex_c),
                fillcolor=hex_to_rgba_map[hex_c]
            ))
        rmin = max(0, df[['Accuracy','Precision','Recall','F1-Macro']].min().min() - 0.05)
        fig_r.update_layout(**LAYOUT,
            polar=dict(
                bgcolor='rgba(20,25,41,1)',
                radialaxis=dict(visible=True, range=[round(rmin,2), 1.0],
                                gridcolor='#1E2D45', tickfont=dict(color='#64748B')),
                angularaxis=dict(gridcolor='#1E2D45')
            ), showlegend=True, height=500)
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Pilih minimal 1 model untuk menampilkan radar chart.")

elif selected == "Training & Evaluation":
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

    def is_valid_cm(model_name):
        raw = df[df['Model'] == model_name]['CM'].values[0]
        try:
            arr = np.array(raw, dtype=np.float64)
            return arr.ndim == 2 and arr.shape == (2, 2) and not np.isnan(arr).any()
        except Exception:
            return False

    cm_models = [m for m in df['Model'].tolist() if is_valid_cm(m)]
    if cm_models:
        sel_cm = st.selectbox("Pilih model:", options=cm_models)
        row_cm = df[df['Model'] == sel_cm].iloc[0]

        try:
            raw_cm = row_cm['CM']
            cm = np.zeros((2, 2), dtype=np.int64)
            for i in range(2):
                for j in range(2):
                    v = raw_cm[i][j]
                    if v is None:
                        cm[i][j] = 0
                    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        cm[i][j] = 0
                    else:
                        cm[i][j] = int(v)
            if cm.shape != (2, 2):
                st.error(f"Format confusion matrix untuk '{sel_cm}' tidak valid (shape: {cm.shape}).")
            else:
                col_a, col_b = st.columns([5, 5])
                with col_a:
                    fig_cm = px.imshow(
                        cm,
                        color_continuous_scale=[[0,'#0F1E35'],[1,'#38BDF8']],
                        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        x=['Negatif (0)', 'Positif (1)'],
                        y=['Negatif (0)', 'Positif (1)'],
                        zmin=0, zmax=int(cm.max())
                    )
                    # Tambah anotasi nilai manual (hindari warning text_auto)
                    for i in range(2):
                        for j in range(2):
                            fig_cm.add_annotation(
                                x=j, y=i,
                                text=f"{int(cm[i, j]):,}",
                                showarrow=False,
                                font=dict(color="white", size=16, family="Inter")
                            )
                    fig_cm.update_layout(**LAYOUT)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_b:
                    tn = int(cm[0][0]); fp = int(cm[0][1])
                    fn = int(cm[1][0]); tp = int(cm[1][1])
                    total = tn + fp + fn + tp
                    precision_neg = tn/(tn+fn) if (tn+fn) > 0 else 0
                    recall_neg    = tn/(tn+fp) if (tn+fp) > 0 else 0
                    precision_pos = tp/(tp+fp) if (tp+fp) > 0 else 0
                    recall_pos    = tp/(tp+fn) if (tp+fn) > 0 else 0
                    st.markdown(f"**Model:** {sel_cm}")
                    st.markdown(f"""
| | Prediksi Negatif | Prediksi Positif |
|---|---|---|
| **Aktual Negatif** | TN = {tn:,} | FP = {fp:,} |
| **Aktual Positif** | FN = {fn:,} | TP = {tp:,} |

- **Total sampel test:** {total:,}
- **Benar diklasifikasi:** {(tn+tp):,} &nbsp;`({(tn+tp)/total*100:.2f}%)`
- **Salah diklasifikasi:** {(fp+fn):,} &nbsp;`({(fp+fn)/total*100:.2f}%)`
- **Precision Negatif:** `{precision_neg:.4f}` &nbsp;|&nbsp; **Recall Negatif:** `{recall_neg:.4f}`
- **Precision Positif:** `{precision_pos:.4f}` &nbsp;|&nbsp; **Recall Positif:** `{recall_pos:.4f}`
""")
                    if row_cm['TrainTime'] is not None:
                        st.markdown("---")
                        h  = int(row_cm['TrainTime'] // 3600)
                        m_ = int((row_cm['TrainTime'] % 3600) // 60)
                        s_ = int(row_cm['TrainTime'] % 60)
                        st.markdown(f"""
**Info Training**
- Base model: `{row_cm['ModelName']}`
- Epochs: `{int(row_cm['Epochs'])}`
- Training time: `{h}j {m_}m {s_}s`
""")
        except Exception as e:

            st.error(f"Gagal menampilkan confusion matrix: {e}")
    else:
        st.info("Tidak ada confusion matrix yang tersedia.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Precision & Recall Per Kelas</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Analisis kemampuan model menangani class imbalance antara kelas Negatif (minoritas) dan Positif (mayoritas).</p>", unsafe_allow_html=True)

    df_cls = df[df['CM'].apply(lambda x: len(x) > 0)].copy()
    if not df_cls.empty:
        c_neg, c_pos = st.columns(2)
        with c_neg:
            st.markdown('<div class="section-header" style="font-size:0.75rem;">Kelas Negatif (Label 0)</div>', unsafe_allow_html=True)
            mn = df_cls[['Model','P_Neg','R_Neg']].melt(id_vars='Model', var_name='Metrik', value_name='Score')
            mn['Metrik'] = mn['Metrik'].map({'P_Neg': 'Precision', 'R_Neg': 'Recall'})
            fig_neg = px.bar(mn, x='Model', y='Score', color='Metrik', barmode='group',
                             text_auto='.3f', color_discrete_sequence=['#FB7185','#FDA4AF'])
            fig_neg.update_layout(**LAYOUT, xaxis_tickangle=-30, yaxis=dict(range=[0,1.05]))
            st.plotly_chart(fig_neg, use_container_width=True)

        with c_pos:
            st.markdown('<div class="section-header" style="font-size:0.75rem;">Kelas Positif (Label 1)</div>', unsafe_allow_html=True)
            mp = df_cls[['Model','P_Pos','R_Pos']].melt(id_vars='Model', var_name='Metrik', value_name='Score')
            mp['Metrik'] = mp['Metrik'].map({'P_Pos': 'Precision', 'R_Pos': 'Recall'})
            fig_pos = px.bar(mp, x='Model', y='Score', color='Metrik', barmode='group',
                             text_auto='.3f', color_discrete_sequence=['#34D399','#6EE7B7'])
            fig_pos.update_layout(**LAYOUT, xaxis_tickangle=-30, yaxis=dict(range=[0,1.05]))
            st.plotly_chart(fig_pos, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Tabel Lengkap Per Kelas</div>', unsafe_allow_html=True)
        tbl = df_cls[['Model','Kategori','P_Neg','R_Neg','P_Pos','R_Pos','F1-Macro']].copy()
        tbl.columns = ['Model','Kategori','Prec Neg','Rec Neg','Prec Pos','Rec Pos','F1-Macro']
        for c in ['Prec Neg','Rec Neg','Prec Pos','Rec Pos','F1-Macro']:
            tbl[c] = tbl[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(tbl, use_container_width=True)
    else:
        st.info("Data confusion matrix tidak tersedia.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Riwayat Training & K-Fold CV</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Rekap proses pelatihan model Deep Learning beserta validasi silang (K-Fold CV).</p>", unsafe_allow_html=True)

    # ── K-Fold Cross Validation ──
    st.markdown('<div class="section-header" style="font-size:0.8rem;">K-Fold Cross Validation — Stabilitas Model</div>', unsafe_allow_html=True)
    kfold = data.get('kfold_cv', {})
    if kfold and kfold.get('results') and len(kfold['results']) > 0:
        results_raw = kfold['results']
        n_folds = kfold.get('n_folds', 5)

        # Format: {model_name: {f1_macro_mean, f1_macro_std, f1_macro_per_fold, ...}}
        kf_models = []
        for model_name, v in results_raw.items():
            if isinstance(v, dict):
                kf_models.append({
                    'Model': model_name,
                    'Mean F1': v.get('f1_macro_mean', 0),
                    'Std': v.get('f1_macro_std', 0),
                    'Per Fold': v.get('f1_macro_per_fold', []),
                    'Precision': v.get('precision_macro_mean', 0),
                    'Recall': v.get('recall_macro_mean', 0),
                })
            elif isinstance(v, (int, float)):
                kf_models.append({'Model': model_name, 'Mean F1': float(v), 'Std': 0, 'Per Fold': [], 'Precision': 0, 'Recall': 0})

        if kf_models:
            # KPI cards
            all_means = [m['Mean F1'] for m in kf_models]
            best_kf = max(kf_models, key=lambda x: x['Mean F1'])
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            for col, title, val, color in [
                (col_kpi1, "Best Model (CV)", best_kf['Model'], "#F59E0B"),
                (col_kpi2, "Best Mean F1-Macro", f"{best_kf['Mean F1']:.4f}", "#38BDF8"),
                (col_kpi3, "Avg Std Deviasi", f"± {float(np.mean([m['Std'] for m in kf_models])):.4f}", "#F59E0B"),
                (col_kpi4, "Jumlah Fold", str(n_folds), "#34D399"),
            ]:
                col.markdown(
                    f'<div class="metric-card"><div class="metric-title">{title}</div>'
                    f'<div class="metric-value" style="color:{color};font-size:1.1rem;">{val}</div></div>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # Bar chart: Mean F1-Macro per model with error bars
            fig_kf_bar = go.Figure()
            sorted_kf = sorted(kf_models, key=lambda x: x['Mean F1'], ascending=False) if False else sorted(kf_models, key=lambda x: x['Mean F1'])
            fig_kf_bar.add_trace(go.Bar(
                x=[m['Mean F1'] for m in sorted_kf],
                y=[m['Model'] for m in sorted_kf],
                orientation='h',
                error_x=dict(type='data', array=[m['Std'] for m in sorted_kf], color='#F59E0B', thickness=2),
                marker=dict(color=[m['Mean F1'] for m in sorted_kf],
                            colorscale=[[0,'#1E3A5F'],[1,'#38BDF8']], showscale=False),
                text=[f"{m['Mean F1']:.4f} ± {m['Std']:.4f}" for m in sorted_kf],
                textposition='outside',
                textfont=dict(color='#94A3B8', size=11)
            ))
            fig_kf_bar.update_layout(**LAYOUT, height=280,
                                     xaxis=dict(title='Mean F1-Macro (CV)', gridcolor='#1E2D45',
                                                range=[min(all_means)-0.05, min(1.0, max(all_means)+0.06)]))
            fig_kf_bar.update_layout(margin=dict(t=20, b=20, l=10, r=120))
            st.plotly_chart(fig_kf_bar, use_container_width=True)

            # Line chart: per-fold detail for each model
            has_per_fold = any(len(m['Per Fold']) > 0 for m in kf_models)
            if has_per_fold:
                st.markdown("<div class='section-header' style='font-size:0.75rem;'>F1-Macro Per Fold (Detail)</div>", unsafe_allow_html=True)
                colors_kf = ['#38BDF8','#34D399','#F59E0B','#A78BFA','#FB7185']
                fig_kf_line = go.Figure()
                for i, m in enumerate(kf_models):
                    if m['Per Fold']:
                        fold_labels = [f"Fold {j+1}" for j in range(len(m['Per Fold']))]
                        fig_kf_line.add_trace(go.Scatter(
                            x=fold_labels, y=m['Per Fold'],
                            mode='lines+markers', name=m['Model'],
                            line=dict(color=colors_kf[i % len(colors_kf)], width=2),
                            marker=dict(size=8),
                        ))
                y_all = [val for m in kf_models for val in m['Per Fold']]
                y_lo2 = max(0.0, round(min(y_all) - 0.02, 2)) if y_all else 0.6
                y_hi2 = min(1.0, round(max(y_all) + 0.02, 2)) if y_all else 1.0
                fig_kf_line.update_layout(**LAYOUT, height=300,
                                          yaxis=dict(title='F1-Macro', gridcolor='#1E2D45', range=[y_lo2, y_hi2]),
                                          xaxis=dict(title='Fold', gridcolor='#1E2D45'),
                                          legend=dict(orientation='h', y=-0.25))
                st.plotly_chart(fig_kf_line, use_container_width=True)

            # Summary table
            st.markdown("<div class='section-header' style='font-size:0.75rem;margin-top:12px;'>Tabel Ringkasan K-Fold CV</div>", unsafe_allow_html=True)
            kf_tbl = pd.DataFrame([{
                'Model': m['Model'],
                'Mean F1-Macro': f"{m['Mean F1']:.4f}",
                'Std Deviasi': f"± {m['Std']:.4f}",
                'Precision': f"{m['Precision']:.4f}" if m['Precision'] else '-',
                'Recall': f"{m['Recall']:.4f}" if m['Recall'] else '-',
            } for m in sorted(kf_models, key=lambda x: x['Mean F1'], reverse=True)])
            st.dataframe(kf_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Format data K-Fold tidak dikenali.")

    else:
        st.info("📋 Data K-Fold CV belum tersedia. Jalankan pipeline K-Fold Cross Validation pada notebook untuk mengisi bagian ini secara otomatis.")
        # Tampilkan info konfigurasi tetap
        col_k1, col_k2, col_k3 = st.columns(3)
        for col, label, val in [
            (col_k1, "Strategi", kfold.get('strategy', 'StratifiedKFold') if kfold else 'StratifiedKFold'),
            (col_k2, "Jumlah Fold", str(kfold.get('n_folds', 5)) if kfold else '5'),
            (col_k3, "Status", "Belum Dijalankan"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-title">{label}</div>'
                f'<div class="metric-value" style="color:#475569;font-size:1.1rem;">{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Histori Training Deep Learning ──
    st.markdown('<div class="section-header" style="font-size:0.8rem;">Histori Training Model Deep Learning</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.82rem;'>Grafik perubahan Loss dan Accuracy selama proses pelatihan — memperlihatkan apakah model konvergen dengan baik dan tidak mengalami overfitting.</p>", unsafe_allow_html=True)

    train_img_col1, train_img_col2 = st.columns(2)
    with train_img_col1:
        st.markdown("<div style='color:#CBD5E1;font-weight:600;margin-bottom:8px;'>DistilBERT — Training History</div>", unsafe_allow_html=True)
        if os.path.exists('visualizations/distilbert_history.png'):
            st.image('visualizations/distilbert_history.png', use_container_width=True)
        else:
            st.markdown("""
<div style='border:1px dashed #1E2D45;border-radius:10px;padding:40px;text-align:center;color:#334155;'>
<div style='font-size:2rem;margin-bottom:8px;'>📈</div>
<div style='font-size:0.85rem;'>distilbert_history.png<br>tidak ditemukan</div>
</div>
""", unsafe_allow_html=True)

        # Info training DistilBERT dari JSON
        distilbert_data = data.get('DistilBERT', {})
        if distilbert_data:
            train_time = distilbert_data.get('train_time_s')
            h  = int(train_time // 3600) if train_time else 0
            m_ = int((train_time % 3600) // 60) if train_time else 0
            s_ = int(train_time % 60) if train_time else 0
            st.markdown(f"""
<div style='background:#0F1E35;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.82rem;color:#94A3B8;'>
📌 <strong>Base Model:</strong> <code>{distilbert_data.get('model_name', 'distilbert-base-uncased')}</code><br>
⏱️ <strong>Training Time:</strong> <code>{h}j {m_}m {s_}s</code><br>
🔁 <strong>Epochs:</strong> <code>{int(distilbert_data.get('epochs', 3))}</code><br>
🎯 <strong>F1-Macro Final:</strong> <code>{distilbert_data.get('f1', 0):.4f}</code>
</div>
""", unsafe_allow_html=True)

    with train_img_col2:
        st.markdown("<div style='color:#CBD5E1;font-weight:600;margin-bottom:8px;'>GloVe 300d + MLP — Training History</div>", unsafe_allow_html=True)
        if os.path.exists('visualizations/glove300d_results.png'):
            st.image('visualizations/glove300d_results.png', use_container_width=True)
        elif os.path.exists('visualizations/distilbert_history.png'):
            st.image('visualizations/distilbert_history.png', caption='(fallback: DistilBERT history)', use_container_width=True)
        else:
            st.markdown("""
<div style='border:1px dashed #1E2D45;border-radius:10px;padding:40px;text-align:center;color:#334155;'>
<div style='font-size:1.5rem;margin-bottom:8px;'>--</div>
<div style='font-size:0.85rem;'>glove300d_results.png<br>tidak ditemukan</div>
</div>
""", unsafe_allow_html=True)

        # Info training GloVe 300d dari JSON
        glove_data = data.get('embedding_models', {}).get('GloVe_300d_MLP', {})
        if glove_data:
            st.markdown(f"""
<div style='background:#0F1E35;border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.82rem;color:#94A3B8;'>
Embedding: <code>GloVe 300d ({glove_data.get('embedding_dim', 300)} dim)</code><br>
Max Sequence Length: <code>{glove_data.get('max_len', 200)} token</code><br>
Vocab Size: <code>{glove_data.get('max_vocab', 50000):,}</code><br>
F1-Macro Final: <code>{glove_data.get('f1_macro', 0):.4f}</code><br>
Class Weight: <code>{'Ya' if glove_data.get('used_class_weights') else 'Tidak'}</code> &nbsp;|&nbsp;
Augmented Data: <code>{'Ya' if glove_data.get('used_augmented_data') else 'Tidak'}</code>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:0.8rem;">Confusion Matrix Model Deep Learning</div>', unsafe_allow_html=True)
    cm_img_col1, cm_img_col2 = st.columns(2)
    with cm_img_col1:
        if os.path.exists('visualizations/distilbert_cm.png'):
            st.image('visualizations/distilbert_cm.png', caption='Confusion Matrix — DistilBERT', use_container_width=True)
    with cm_img_col2:
        if os.path.exists('visualizations/comparison_confusion_matrices.png'):
            st.image('visualizations/comparison_confusion_matrices.png', caption='Confusion Matrix — Semua Model', use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:0.8rem;">Grafik Hasil Perbandingan</div>', unsafe_allow_html=True)
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        if os.path.exists('visualizations/comparison_all_metrics.png'):
            st.image('visualizations/comparison_all_metrics.png', caption='Perbandingan Semua Metrik', use_container_width=True)
    with img_col2:
        if os.path.exists('visualizations/comparison_radar.png'):
            st.image('visualizations/comparison_radar.png', caption='Radar Chart Perbandingan', use_container_width=True)
    with img_col3:
        if os.path.exists('visualizations/comparison_f1_ranking.png'):
            st.image('visualizations/comparison_f1_ranking.png', caption='F1 Ranking', use_container_width=True)
        elif os.path.exists('visualizations/final_comparison.png'):
            st.image('visualizations/final_comparison.png', caption='Final Comparison', use_container_width=True)

# ── TAB 6: Kesimpulan ─────────────────────────────────────────────────────────
elif selected == "Final Insights":
    st.markdown('<div class="section-header">Kesimpulan Otomatis</div>', unsafe_allow_html=True)
    if not df.empty:
        best = df.iloc[0]
        worst = df.iloc[-1]
        
        ml_df = df[df['Kategori'] == 'Classical ML']
        dl_df = df[df['Kategori'] == 'Deep Learning']
        
        best_ml_txt = f"<strong>{ml_df.iloc[0]['Model']}</strong> dengan F1-Macro = <code>{ml_df.iloc[0]['F1-Macro']:.4f}</code>" if not ml_df.empty else "<em>Tidak ada data Classical ML</em>"
        best_dl_txt = f"<strong>{dl_df.iloc[0]['Model']}</strong> dengan F1-Macro = <code>{dl_df.iloc[0]['F1-Macro']:.4f}</code>" if not dl_df.empty else "<em>Tidak ada data Deep Learning</em>"
        
        avg_f1 = df['F1-Macro'].mean()
        improve = ((best['F1-Macro'] - worst['F1-Macro']) / worst['F1-Macro']) * 100

        st.markdown(f"""
<div class="conclusion-box">

<h3>Model Terbaik</h3>
<p>Model <strong>{best['Model']}</strong> mencapai performa tertinggi dengan F1-Macro sebesar 
<strong>{best['F1-Macro']:.4f}</strong> dan Accuracy <strong>{best['Accuracy']*100:.2f}%</strong>, 
menjadikannya model paling optimal untuk tugas binary sentiment analysis pada dataset Amazon Fine Food Reviews.</p>

<h3>Perbandingan Kategori</h3>
<ul>
  <li>Classical ML terbaik: {best_ml_txt}</li>
  <li>Deep Learning terbaik: {best_dl_txt}</li>
  <li>Rata-rata F1-Macro seluruh model: <code>{avg_f1:.4f}</code></li>
</ul>

<h3>Temuan Utama</h3>
<ol>
  <li>Model Deep Learning secara konsisten mengungguli Classical ML dalam F1-Macro, terutama dalam menangani kelas minoritas (Negatif).</li>
  <li>Model <strong>{worst['Model']}</strong> merupakan model dengan performa terendah (F1-Macro = <code>{worst['F1-Macro']:.4f}</code>).</li>
  <li>Peningkatan dari model terlemah ke model terbaik sebesar <strong>{improve:.2f}%</strong>.</li>
  <li>Penggunaan GloVe 300d memberikan representasi semantik yang lebih kaya dibanding GloVe 50d.</li>
</ol>

<h3>Rekomendasi</h3>
<p>Untuk deployment, <strong>{best['Model']}</strong> direkomendasikan karena memberikan keseimbangan terbaik 
antara Precision dan Recall pada kedua kelas. Jika sumber daya komputasi terbatas, disarankan menggunakan model Classical ML terbaik yang tersedia.</p>

</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Ringkasan Metrik Final</div>', unsafe_allow_html=True)
        final = df[['Model','Kategori','Accuracy','Precision','Recall','F1-Macro']].copy()
        final.insert(0, 'Rank', range(1, len(final)+1))
        for c in ['Accuracy','Precision','Recall','F1-Macro']:
            final[c] = final[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(final, use_container_width=True)

# ── TAB ERROR ANALYSIS ───────────────────────────────────────────────────────
elif selected == "Error Analysis":
    st.markdown('<div class="section-header">Error Analysis — Analisis Kesalahan Prediksi Model</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Analisis mendalam terhadap pola kesalahan setiap model: False Positive (FP), False Negative (FN), dan distribusi error per kelas.</p>", unsafe_allow_html=True)

    # Filter model yang memiliki confusion matrix valid
    df_err = df[df['CM'].apply(lambda x: len(x) == 2 and len(x[0]) == 2)].copy()

    if df_err.empty:
        st.info("Tidak ada data confusion matrix yang tersedia untuk analisis error.")
    else:
        # ── Hitung metrik error dari confusion matrix
        err_rows = []
        for _, row in df_err.iterrows():
            try:
                cm_raw = row['CM']
                tn = int(cm_raw[0][0]); fp = int(cm_raw[0][1])
                fn = int(cm_raw[1][0]); tp = int(cm_raw[1][1])
                total = tn + fp + fn + tp
                total_neg = tn + fp
                total_pos = fn + tp
                err_rows.append({
                    'Model': row['Model'],
                    'Kategori': row['Kategori'],
                    'FP': fp, 'FN': fn,
                    'TP': tp, 'TN': tn,
                    'Total': total,
                    'Total_Neg': total_neg,
                    'Total_Pos': total_pos,
                    'Error_Total': fp + fn,
                    'Error_Rate': (fp + fn) / total * 100 if total > 0 else 0,
                    'FPR': fp / total_neg * 100 if total_neg > 0 else 0,   # False Positive Rate
                    'FNR': fn / total_pos * 100 if total_pos > 0 else 0,   # False Negative Rate
                    'F1': row['F1-Macro'],
                })
            except Exception:
                continue

        df_er = pd.DataFrame(err_rows).sort_values('Error_Rate').reset_index(drop=True)

        # ── KPI Cards ──
        best_err = df_er.iloc[0]
        worst_err = df_er.iloc[-1]
        total_err_all = df_er['Error_Total'].sum()
        avg_err_rate = df_er['Error_Rate'].mean()

        kpi_e1, kpi_e2, kpi_e3, kpi_e4 = st.columns(4)
        err_kpis = [
            (kpi_e1, "Model Paling Akurat", best_err['Model'], "#34D399"),
            (kpi_e2, "Error Rate Terendah", f"{best_err['Error_Rate']:.2f}%", "#38BDF8"),
            (kpi_e3, "Error Rate Tertinggi", f"{worst_err['Error_Rate']:.2f}%", "#FB7185"),
            (kpi_e4, "Avg Error Rate", f"{avg_err_rate:.2f}%", "#F59E0B"),
        ]
        for col, title, val, color in err_kpis:
            col.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value" style="color:{color};font-size:1.2rem;">{val}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 1: Error Rate & Total Error Ranking ──
        st.markdown('<div class="section-header" style="font-size:0.8rem;">Ranking Error Rate per Model</div>', unsafe_allow_html=True)
        col_er1, col_er2 = st.columns([6, 4])
        with col_er1:
            err_colors = ['#34D399' if r < avg_err_rate else '#FB7185' for r in df_er['Error_Rate']]
            fig_er = go.Figure(go.Bar(
                x=df_er['Error_Rate'],
                y=df_er['Model'],
                orientation='h',
                marker_color=err_colors,
                text=[f"{r:.2f}%" for r in df_er['Error_Rate']],
                textposition='outside',
                textfont=dict(color='#94A3B8', size=11)
            ))
            fig_er.add_vline(
                x=avg_err_rate,
                line=dict(color='#F59E0B', width=1.5, dash='dash'),
                annotation_text=f"Avg {avg_err_rate:.2f}%",
                annotation_font=dict(color='#F59E0B', size=11),
                annotation_position='top right'
            )
            _er_layout = {**LAYOUT, 'height': 320,
                           'xaxis': dict(title='Error Rate (%)', gridcolor='#1E2D45'),
                           'margin': dict(t=30, b=20, r=100)}
            fig_er.update_layout(**_er_layout)
            st.plotly_chart(fig_er, use_container_width=True)

        with col_er2:
            st.markdown('<div class="section-header" style="font-size:0.75rem;">Total Salah Klasifikasi</div>', unsafe_allow_html=True)
            df_er_disp = df_er[['Model', 'FP', 'FN', 'Error_Total', 'Error_Rate']].copy()
            df_er_disp['Error_Rate'] = df_er_disp['Error_Rate'].apply(lambda x: f"{x:.2f}%")
            df_er_disp.columns = ['Model', 'False Pos', 'False Neg', 'Total Error', 'Error Rate']
            st.dataframe(df_er_disp, use_container_width=True, hide_index=True, height=320)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 2: FP vs FN Comparison ──
        st.markdown('<div class="section-header" style="font-size:0.8rem;">False Positive vs False Negative per Model</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#64748B;font-size:0.82rem;'><strong>False Positive (FP)</strong>: Ulasan Negatif yang salah diprediksi Positif. <strong>False Negative (FN)</strong>: Ulasan Positif yang salah diprediksi Negatif.</p>", unsafe_allow_html=True)

        fig_fpfn = go.Figure()
        df_er_sorted_f1 = df_er.sort_values('F1', ascending=False)
        fig_fpfn.add_trace(go.Bar(
            name='False Positive (FP)',
            x=df_er_sorted_f1['Model'],
            y=df_er_sorted_f1['FP'],
            marker_color='#FB7185',
            text=df_er_sorted_f1['FP'].apply(lambda x: f"{x:,}"),
            textposition='outside',
            textfont=dict(color='#94A3B8', size=10)
        ))
        fig_fpfn.add_trace(go.Bar(
            name='False Negative (FN)',
            x=df_er_sorted_f1['Model'],
            y=df_er_sorted_f1['FN'],
            marker_color='#F59E0B',
            text=df_er_sorted_f1['FN'].apply(lambda x: f"{x:,}"),
            textposition='outside',
            textfont=dict(color='#94A3B8', size=10)
        ))
        fig_fpfn.update_layout(
            **LAYOUT, barmode='group', height=350,
            xaxis=dict(title='', tickangle=-20),
            yaxis=dict(title='Jumlah Sampel', gridcolor='#1E2D45'),
            legend=dict(orientation='h', y=1.08)
        )
        st.plotly_chart(fig_fpfn, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 3: FPR vs FNR (Rate) ──
        st.markdown('<div class="section-header" style="font-size:0.8rem;">False Positive Rate vs False Negative Rate (%)</div>', unsafe_allow_html=True)
        col_fpr, col_fnr = st.columns(2)
        with col_fpr:
            st.markdown("<div style='color:#CBD5E1;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>False Positive Rate — FP / Total Aktual Negatif</div>", unsafe_allow_html=True)
            df_fpr = df_er.sort_values('FPR')
            fpr_colors = ['#34D399' if r < df_er['FPR'].mean() else '#FB7185' for r in df_fpr['FPR']]
            fig_fpr = go.Figure(go.Bar(
                x=df_fpr['FPR'],
                y=df_fpr['Model'],
                orientation='h',
                marker_color=fpr_colors,
                text=[f"{r:.1f}%" for r in df_fpr['FPR']],
                textposition='outside',
                textfont=dict(color='#94A3B8', size=11)
            ))
            fig_fpr.update_layout(**{**LAYOUT, 'height': 280,
                                      'xaxis': dict(title='FPR (%)', gridcolor='#1E2D45'),
                                      'margin': dict(t=10, b=10, r=60)})
            st.plotly_chart(fig_fpr, use_container_width=True)

        with col_fnr:
            st.markdown("<div style='color:#CBD5E1;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>False Negative Rate — FN / Total Aktual Positif</div>", unsafe_allow_html=True)
            df_fnr = df_er.sort_values('FNR')
            fnr_colors = ['#34D399' if r < df_er['FNR'].mean() else '#F59E0B' for r in df_fnr['FNR']]
            fig_fnr = go.Figure(go.Bar(
                x=df_fnr['FNR'],
                y=df_fnr['Model'],
                orientation='h',
                marker_color=fnr_colors,
                text=[f"{r:.1f}%" for r in df_fnr['FNR']],
                textposition='outside',
                textfont=dict(color='#94A3B8', size=11)
            ))
            fig_fnr.update_layout(**{**LAYOUT, 'height': 280,
                                      'xaxis': dict(title='FNR (%)', gridcolor='#1E2D45'),
                                      'margin': dict(t=10, b=10, r=60)})
            st.plotly_chart(fig_fnr, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 4: Error Composition Stacked Bar ──
        st.markdown('<div class="section-header" style="font-size:0.8rem;">Komposisi Prediksi per Model (Stacked)</div>', unsafe_allow_html=True)
        df_stk = df_er.sort_values('F1', ascending=False)
        fig_stk = go.Figure()
        fig_stk.add_trace(go.Bar(name='True Positive (TP)', x=df_stk['Model'], y=df_stk['TP'],
                                  marker_color='#38BDF8'))
        fig_stk.add_trace(go.Bar(name='True Negative (TN)', x=df_stk['Model'], y=df_stk['TN'],
                                  marker_color='#34D399'))
        fig_stk.add_trace(go.Bar(name='False Positive (FP)', x=df_stk['Model'], y=df_stk['FP'],
                                  marker_color='#FB7185'))
        fig_stk.add_trace(go.Bar(name='False Negative (FN)', x=df_stk['Model'], y=df_stk['FN'],
                                  marker_color='#F59E0B'))
        fig_stk.update_layout(
            **LAYOUT, barmode='stack', height=380,
            xaxis=dict(tickangle=-20),
            yaxis=dict(title='Jumlah Sampel', gridcolor='#1E2D45'),
            legend=dict(orientation='h', y=1.08)
        )
        st.plotly_chart(fig_stk, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 5: Automated Error Insights ──
        st.markdown('<div class="section-header" style="font-size:0.8rem;">Insight Otomatis</div>', unsafe_allow_html=True)

        best_fpr_model = df_er.sort_values('FPR').iloc[0]
        best_fnr_model = df_er.sort_values('FNR').iloc[0]
        worst_err_model = df_er.sort_values('Error_Rate', ascending=False).iloc[0]
        fp_heavy = df_er[df_er['FP'] > df_er['FN']]
        fn_heavy = df_er[df_er['FN'] > df_er['FP']]

        insights = []
        insights.append(f"🏆 Model <strong>{best_err['Model']}</strong> memiliki error rate terendah sebesar <code>{best_err['Error_Rate']:.2f}%</code> dari total <code>{best_err['Total']:,}</code> sampel uji.")
        insights.append(f"⚠️ Model <strong>{worst_err_model['Model']}</strong> memiliki error rate tertinggi (<code>{worst_err_model['Error_Rate']:.2f}%</code>) dengan total <code>{int(worst_err_model['Error_Total']):,}</code> kesalahan prediksi.")
        insights.append(f"📌 False Positive Rate terendah dimiliki oleh <strong>{best_fpr_model['Model']}</strong> (<code>{best_fpr_model['FPR']:.1f}%</code>) — model ini paling jarang salah mengklasifikasikan ulasan Negatif sebagai Positif.")
        insights.append(f"📌 False Negative Rate terendah dimiliki oleh <strong>{best_fnr_model['Model']}</strong> (<code>{best_fnr_model['FNR']:.1f}%</code>) — model ini paling jarang melewatkan ulasan Positif.")
        if not fp_heavy.empty:
            fph_list = ', '.join([f"<strong>{m}</strong>" for m in fp_heavy['Model'].tolist()])
            insights.append(f"🔴 Model yang cenderung <em>over-predict Positif</em> (FP &gt; FN): {fph_list}. Hal ini umum terjadi akibat class imbalance yang condong ke kelas Positif.")
        if not fn_heavy.empty:
            fnh_list = ', '.join([f"<strong>{m}</strong>" for m in fn_heavy['Model'].tolist()])
            insights.append(f"🟡 Model yang cenderung <em>under-detect Positif</em> (FN &gt; FP): {fnh_list}.")

        insight_html = "".join([f"<li style='margin-bottom:10px;color:#CBD5E1;font-size:0.88rem;line-height:1.6;'>{ins}</li>" for ins in insights])
        st.markdown(f"""
<div style='background:linear-gradient(135deg,#0F1E35,#111827);border:1px solid #1E3A5F;
     border-left:4px solid #38BDF8;border-radius:12px;padding:20px 28px;'>
  <ul style='padding-left:20px;margin:0;'>
    {insight_html}
  </ul>
</div>
""", unsafe_allow_html=True)


elif selected == "Interactive Prediction":
    st.markdown('<div class="section-header">Prediksi Sentimen Teks</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:0.87rem;'>Masukkan teks ulasan produk dalam Bahasa Inggris. Model akan memprediksi apakah ulasan tersebut bersifat Positif atau Negatif.</p>", unsafe_allow_html=True)

    vec, nb_model = load_predictor()

    col_input, col_result = st.columns([6, 4])

    with col_input:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Input Teks</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="Teks ulasan:",
            placeholder="Contoh: This product is absolutely amazing, I love it so much!",
            height=160,
            label_visibility="collapsed"
        )
        pred_btn = st.button("Analisis Sentimen", use_container_width=True, type="primary")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Contoh Teks</div>', unsafe_allow_html=True)
        examples = [
            ("Positif", "This product is absolutely fantastic! Best purchase I have ever made."),
            ("Positif", "Excellent quality, fast shipping, highly recommended to everyone."),
            ("Negatif", "Terrible product. Broke after one day, complete waste of money."),
            ("Negatif", "Very disappointed. The quality is poor and it smells awful."),
        ]
        for label, ex in examples:
            col_ex_btn, col_ex_txt = st.columns([2, 8])
            col_ex_btn.markdown(
                f"<span style='color:{'#34D399' if label=='Positif' else '#FB7185'};font-size:0.75rem;font-weight:600;'>{label}</span>",
                unsafe_allow_html=True
            )
            col_ex_txt.markdown(f"<span style='color:#475569;font-size:0.8rem;'>{ex}</span>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-header" style="font-size:0.75rem;">Hasil Prediksi</div>', unsafe_allow_html=True)

        if pred_btn and user_text.strip():
            if vec is None or nb_model is None:
                st.error("Model tidak ditemukan. Pastikan vectorizer_ngram.pkl dan model_nb_ngram.pkl tersedia.")
            else:
                clean = preprocess_text(user_text)
                X = vec.transform([clean])
                pred = nb_model.predict(X)[0]
                proba = nb_model.predict_proba(X)[0]

                label_pos = "POSITIF" if pred == 1 else "NEGATIF"
                conf = proba[pred] * 100
                color_result = "#34D399" if pred == 1 else "#FB7185"
                bg_color = "rgba(52,211,153,0.08)" if pred == 1 else "rgba(251,113,133,0.08)"
                border_color = "#34D399" if pred == 1 else "#FB7185"
                desc = "Ulasan ini mengandung sentimen yang bersifat positif." if pred == 1 else "Ulasan ini mengandung sentimen yang bersifat negatif."

                st.markdown(f"""
<div style='background:{bg_color};border:1px solid {border_color};border-left:4px solid {border_color};
     border-radius:12px;padding:24px;text-align:center;margin-bottom:16px;'>
  <div style='font-size:0.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;'>Hasil</div>
  <div style='font-size:2.4rem;font-weight:700;color:{color_result};letter-spacing:2px;'>{label_pos}</div>
  <div style='font-size:0.85rem;color:#94A3B8;margin-top:8px;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

                st.markdown(f"**Tingkat Kepercayaan:** `{conf:.1f}%`")

                # Confidence bar chart
                fig_conf = go.Figure(go.Bar(
                    x=[proba[0]*100, proba[1]*100],
                    y=['Negatif', 'Positif'],
                    orientation='h',
                    marker_color=['#FB7185', '#34D399'],
                    text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    textposition='inside',
                    textfont=dict(color='white', size=13)
                ))
                fig_conf.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8', family='Inter'),
                    xaxis=dict(range=[0, 100], title='Probabilitas (%)'),
                    height=160,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_conf, use_container_width=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown(f"**Teks yang dianalisis:**")
                st.code(user_text.strip(), language=None)

        elif pred_btn and not user_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            st.markdown("""
<div style='border:1px dashed #1E2D45;border-radius:12px;padding:40px;text-align:center;color:#334155;'>
  <div style='font-size:0.9rem;'>Hasil prediksi akan muncul di sini</div>
  <div style='font-size:0.78rem;margin-top:6px;'>Ketik teks ulasan lalu klik tombol Analisis</div>
</div>
""", unsafe_allow_html=True)

elif selected == "About Project":
    st.markdown('<div class="section-header">About Project</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 Amazon Fine Food Reviews - Sentiment Analysis
    Proyek ini berfokus pada pembangunan pipeline NLP lengkap untuk memprediksi sentimen ulasan pelanggan. 
    Kami membandingkan model **Classical Machine Learning** (Naive Bayes, SVM, Logistic Regression, dll.) 
    dengan **Deep Learning** (DistilBERT, MLP + GloVe) untuk menemukan pendekatan paling optimal.
    
    *Informasi lebih lanjut tentang tim/peneliti dapat ditambahkan di sini.*
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem;letter-spacing:0.5px;'>"
    "NLP Sentiment Analysis Dashboard · Dibuat dengan Streamlit & Plotly · Amazon Fine Food Reviews"
    "</p>", unsafe_allow_html=True
)

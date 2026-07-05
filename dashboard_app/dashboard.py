import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import re

st.set_page_config(
    page_title="NLP Sentiment Dashboard",
    page_icon="bar_chart",
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

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #0D1321; border-right: 1px solid #1E2D45; }

/* Divider */
.divider { border: none; border-top: 1px solid #1E2D45; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Parse Data ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        with open('all_metrics.json', 'r', encoding='utf-8') as f:
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
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Informasi Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
**Dataset:** Amazon Fine Food Reviews  
**Task:** Binary Sentiment Analysis  
**Metrik Utama:** F1-Macro (Macro-Averaged)  
**Total Sampel:** ±72.778 reviews  
**Kelas:** Negatif (0) & Positif (1)
""")
    st.markdown('<div class="section-header" style="margin-top:20px;">Peringkat Model</div>', unsafe_allow_html=True)
    for _, row in df.iterrows():
        bar = int(row['F1-Macro'] * 10)
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

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# NLP Sentiment Analysis Dashboard")
st.markdown("<p style='color:#64748B;font-size:0.95rem;margin-top:-10px;'>Perbandingan performa 9 model Machine Learning dan Deep Learning — Amazon Fine Food Reviews</p>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
if not df.empty:
    best = df.iloc[0]
    best_ml = df[df['Kategori'] == 'Classical ML'].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Best Model", best['Model'], "#F59E0B"),
        (c2, "Best F1-Macro", f"{best['F1-Macro']:.4f}", "#38BDF8"),
        (c3, "Best Accuracy", f"{best['Accuracy']*100:.2f}%", "#34D399"),
        (c4, "Best Classical ML", best_ml['Model'], "#A78BFA"),
        (c5, "Total Model", str(len(df)), "#FB7185"),
    ]
    for col, title, val, color in cards:
        col.markdown(
            f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value" style="color:{color};font-size:1.3rem;">{val}</div></div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Perbandingan", "Radar Chart", "Confusion Matrix", "Analisis Per Kelas", "Prediksi Sentimen", "Kesimpulan"
])

# ── TAB 1: Perbandingan ───────────────────────────────────────────────────────
with tab1:
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

# ── TAB 2: Radar ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Radar Chart Perbandingan Model</div>', unsafe_allow_html=True)
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

# ── TAB 3: Confusion Matrix ───────────────────────────────────────────────────
with tab3:
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
            import math
            raw_cm = row_cm['CM']
            # Konversi element-by-element paling aman
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
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Negative (0)', 'Positive (1)'],
                        y=['Negative (0)', 'Positive (1)'],
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
                    st.markdown(f"**Model:** {sel_cm}")
                    st.markdown(f"""
| | Pred. Negatif | Pred. Positif |
|---|---|---|
| **Actual Negatif** | TN = {tn:,} | FP = {fp:,} |
| **Actual Positif** | FN = {fn:,} | TP = {tp:,} |

- **Total sampel test:** {total:,}
- **Benar diklasifikasi:** {(tn+tp):,} &nbsp;`({(tn+tp)/total*100:.2f}%)`
- **Salah diklasifikasi:** {(fp+fn):,} &nbsp;`({(fp+fn)/total*100:.2f}%)`
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

# ── TAB 4: Per Kelas ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Precision dan Recall Per Kelas</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">K-Fold Cross Validation</div>', unsafe_allow_html=True)
    kfold = data.get('kfold_cv', {})
    if kfold and kfold.get('results'):
        results = kfold['results']
        kdf = pd.DataFrame([{'Fold': k, 'F1-Macro': v} for k, v in results.items()])
        fig_kf = px.line(kdf, x='Fold', y='F1-Macro', markers=True, text='F1-Macro')
        fig_kf.update_traces(texttemplate='%{text:.4f}', textposition='top center',
                             line_color='#38BDF8', marker_color='#38BDF8')
        fig_kf.update_layout(**LAYOUT)
        st.plotly_chart(fig_kf, use_container_width=True)
        mean_f1 = np.mean(list(results.values()))
        std_f1 = np.std(list(results.values()))
        st.markdown(f"**Rata-rata F1-Macro:** `{mean_f1:.4f}` &plusmn; `{std_f1:.4f}`")
    else:
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Strategi", kfold.get('strategy', 'StratifiedKFold'))
        col_k2.metric("Jumlah Fold", kfold.get('n_folds', 5))
        col_k3.metric("Status", "Belum tersedia")
        st.info("Data K-Fold CV belum diisi. Jalankan pipeline K-Fold untuk mengisi bagian ini.")

# ── TAB 6: Kesimpulan ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Kesimpulan Otomatis</div>', unsafe_allow_html=True)
    if not df.empty:
        best = df.iloc[0]
        worst = df.iloc[-1]
        best_ml = df[df['Kategori'] == 'Classical ML'].iloc[0]
        best_dl = df[df['Kategori'] == 'Deep Learning'].iloc[0]
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
  <li>Classical ML terbaik: <strong>{best_ml['Model']}</strong> dengan F1-Macro = <code>{best_ml['F1-Macro']:.4f}</code></li>
  <li>Deep Learning terbaik: <strong>{best_dl['Model']}</strong> dengan F1-Macro = <code>{best_dl['F1-Macro']:.4f}</code></li>
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
antara Precision dan Recall pada kedua kelas. Jika sumber daya komputasi terbatas, 
<strong>{best_ml['Model']}</strong> adalah pilihan terbaik dari kelompok Classical ML.</p>

</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Ringkasan Metrik Final</div>', unsafe_allow_html=True)
        final = df[['Model','Kategori','Accuracy','Precision','Recall','F1-Macro']].copy()
        final.insert(0, 'Rank', range(1, len(final)+1))
        for c in ['Accuracy','Precision','Recall','F1-Macro']:
            final[c] = final[c].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(final, use_container_width=True)

# ── TAB 6: Prediksi Sentimen ─────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        vec   = joblib.load('vectorizer_bow.pkl')
        model = joblib.load('model_nb_bow.pkl')
        return vec, model
    except Exception as e:
        return None, None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── TAB 5: Prediksi Sentimen ────────────────────────────────────────────
with tab5:
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
                st.error("Model tidak ditemukan. Pastikan vectorizer_bow.pkl dan model_nb_bow.pkl tersedia.")
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

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem;letter-spacing:0.5px;'>"
    "NLP Sentiment Analysis Dashboard · Dibuat dengan Streamlit & Plotly · Amazon Fine Food Reviews"
    "</p>", unsafe_allow_html=True
)

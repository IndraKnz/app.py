import io
from datetime import datetime

import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.preprocessing import preprocess_series
from utils.model import train_model, predict_text, evaluate_model, get_top_tfidf_words
from utils.aspect import determine_aspects, ASPECTS


st.set_page_config(
    page_title="Analisa Sentimen: Layanan Akademik",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .stApp { background: #f5f7fb; color: #172033; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }
    .hero h1 { margin: 0; font-size: 2.25rem; font-weight: 800; color: #0f172a; }
    .hero p { margin-top: .2rem; color: #526071; font-size: 1rem; }
    .card {
        background: #ffffff; border: 1px solid #dfe6ef; border-radius: 14px;
        padding: 18px; box-shadow: 0 4px 16px rgba(15,23,42,.06);
        height: 100%;
    }
    .nav-note {
        background: #0f355e; color: white; padding: 10px 16px;
        border-radius: 12px; margin-bottom: 12px;
    }
    .metric-card {
        background: #fff; border: 1px solid #dfe6ef; border-radius: 14px;
        padding: 18px; box-shadow: 0 4px 16px rgba(15,23,42,.05);
        text-align: center;
    }
    .metric-title { color:#5c6878; font-size:.9rem; }
    .metric-value { color:#0f355e; font-size:2rem; font-weight:800; margin-top:4px; }
    .result-positive {
        background:#e7f7ec; border:1px solid #b8e5c4; color:#18713a;
        border-radius:10px; padding:12px 15px; font-weight:700;
    }
    .result-negative {
        background:#fdecec; border:1px solid #f1bcbc; color:#a12a2a;
        border-radius:10px; padding:12px 15px; font-weight:700;
    }
    .result-neutral {
        background:#eef1f5; border:1px solid #d4d9e1; color:#4b5563;
        border-radius:10px; padding:12px 15px; font-weight:700;
    }
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #dfe6ef; border-radius: 14px;
        padding: 14px;
    }
    .small-muted { color:#687587; font-size:.85rem; }

    .dash-card { background:#fff; border:1px solid #dfe6ef; border-radius:14px; padding:14px; box-shadow:0 4px 16px rgba(15,23,42,.06); min-height:100%; }
    .dash-card h3 { margin:0 0 12px 0; font-size:1.05rem; color:#111827; }
    .dark-card { background:linear-gradient(145deg,#123e70,#0b2748); color:#fff; border:none; }
    .dark-card h3 { color:#fff; }
    .summary-item { display:flex; align-items:center; gap:10px; margin:16px 0; }
    .summary-icon { width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; background:#fff; color:#17395f; }
    .summary-icon.pos { background:#dff4e6; } .summary-icon.neg { background:#fde1e1; } .summary-icon.neu { background:#e8edf3; }
    .summary-label { font-size:.75rem; opacity:.9; }
    .summary-value { font-size:1.18rem; font-weight:800; } .summary-value span { font-size:.8rem; font-weight:500; opacity:.9; }
    .mini-metric { background:linear-gradient(145deg,#edf7ff,#dcecf9); border:1px solid #c9ddeb; border-radius:12px; padding:14px 8px; margin:8px 0; text-align:center; }
    .mini-value { font-size:1.75rem; font-weight:800; color:#07142d; letter-spacing:-.03em; }
</style>
""", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def make_unique_columns(columns):
    """Return unique, cleaned column names without losing the original order."""
    seen = {}
    result = []
    for col in columns:
        name = str(col).strip()
        if not name:
            name = "Kolom"
        count = seen.get(name, 0)
        result.append(name if count == 0 else f"{name}_{count+1}")
        seen[name] = count + 1
    return result


def read_csv_safely(uploaded_file):
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("File CSV kosong.")

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            df.columns = make_unique_columns(df.columns)
            if df.shape[1] == 0:
                raise ValueError("CSV tidak memiliki kolom.")
            return df
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError("CSV tidak dapat dibaca. Pastikan delimiter dan encoding file benar.")


def label_display(label):
    value = str(label).strip().lower()
    mapping = {
        "positif": "Positif", "positive": "Positif", "pos": "Positif", "+": "Positif",
        "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif", "-": "Negatif",
        "netral": "Netral", "neutral": "Netral", "neu": "Netral", "0": "Netral",
    }
    return mapping.get(value, str(label).strip().title())


def sentiment_counts(series):
    display = series.map(label_display)
    order = ["Positif", "Negatif", "Netral"]
    counts = display.value_counts()
    return pd.Series({k: int(counts.get(k, 0)) for k in order if counts.get(k, 0) > 0})


def aspect_rows(df):
    rows = []
    for _, row in df.iterrows():
        aspects = row.get("Aspek", [])
        if not isinstance(aspects, list):
            aspects = [aspects] if pd.notna(aspects) else []
        for aspect in aspects:
            if aspect in ASPECTS:
                rows.append({
                    "Aspek": aspect,
                    "Sentimen": label_display(row["Sentimen Prediksi"]),
                    "Ulasan": row["Ulasan"],
                })
    return pd.DataFrame(rows)


def format_pct(value):
    return f"{value * 100:.2f}%"


# =========================
# SESSION STATE
# =========================
for key, default in {
    "dataset": None,
    "analysis_ready": False,
    "uploaded_name": None,
    "models": None,
    "metrics": None,
    "test_data": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
<h1>Analisa Sentimen: Layanan Akademik</h1>
<p>Metode Naive Bayes + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INITIAL UPLOAD
# =========================
if st.session_state.dataset is None:
    st.info("Upload dataset CSV terlebih dahulu. Seluruh statistik, grafik, model, dan hasil analisis akan dihitung dari dataset yang Anda upload.")
    uploaded = st.file_uploader("Upload Dataset CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = read_csv_safely(uploaded)
            if df.empty:
                st.error("CSV berhasil dibaca tetapi tidak memiliki baris data.")
                st.stop()

            st.success(f"Dataset berhasil dibaca: {df.shape[0]:,} baris × {df.shape[1]:,} kolom.")
            c1, c2 = st.columns(2)
            c1.metric("Jumlah Baris", f"{len(df):,}")
            c2.metric("Jumlah Kolom", f"{len(df.columns):,}")

            st.write("**Nama kolom:**", ", ".join(map(str, df.columns)))
            st.caption("Preview di bawah menampilkan seluruh kolom asli dari CSV. Aplikasi tidak menambahkan kolom rating atau kolom lain yang tidak ada di file.")
            st.dataframe(df, width='stretch', height=360, hide_index=True)

            st.divider()
            st.subheader("Pemetaan Kolom Dataset")

            columns = list(df.columns)
            lower_map = {str(c).lower().strip(): c for c in columns}

            # Hanya pilih default jika nama kolom memang cocok.
            # Jangan pernah menggunakan kolom pertama sebagai fallback untuk label
            # dan jangan pernah menganggap kolom sembarang sebagai tanggal.
            def guess(candidates):
                for candidate in candidates:
                    if candidate in lower_map:
                        return lower_map[candidate]
                return None

            default_text = guess([
                "ulasan", "text", "teks", "komentar", "review", "comment",
                "tweet", "feedback", "opini", "pertanyaan"
            ])
            default_label = guess([
                "sentimen", "sentiment", "label", "kategori", "class", "kelas"
            ])

            # Kolom tanggal hanya boleh muncul jika nama kolomnya memang
            # mengindikasikan tanggal/waktu. Kolom seperti rating tidak akan
            # pernah otomatis masuk ke pemetaan tanggal.
            date_names = {
                "tanggal", "tgl", "date", "waktu", "time", "datetime",
                "timestamp", "created_at", "createdat", "tanggal_ulasan",
                "waktu_ulasan"
            }
            date_candidates = [
                c for c in columns
                if str(c).strip().lower().replace(" ", "_") in date_names
            ]
            date_options = ["(Tidak ada)"] + date_candidates
            default_date = date_candidates[0] if date_candidates else "(Tidak ada)"

            if default_text is None:
                st.warning("Kolom ulasan belum dapat dikenali otomatis. Silakan pilih kolom ulasan secara manual.")
            if default_label is None:
                st.warning("Kolom sentimen/label belum dapat dikenali otomatis. Silakan pilih kolom sentimen secara manual.")

            text_options = columns
            text_index = text_options.index(default_text) if default_text in text_options else 0
            text_col = st.selectbox(
                "Kolom Ulasan", text_options, index=text_index
            )

            label_options = ["(Tidak ada)"] + columns
            label_index = (columns.index(default_label) + 1) if default_label in columns else 0
            label_col = st.selectbox(
                "Kolom Sentimen/Label", label_options, index=label_index
            )

            date_col = st.selectbox(
                "Kolom Tanggal (opsional)", date_options,
                index=date_options.index(default_date)
            )

            if label_col == "(Tidak ada)":
                st.warning("Kolom sentimen/label belum dipilih. Supervised learning Multinomial Naive Bayes membutuhkan label sentimen.")
            elif st.button("🚀 Mulai Analisis", type="primary", width='stretch'):
                try:
                    result = df.copy()
                    # Normalisasi ke nama internal yang stabil. Jangan memakai nama
                    # kolom asli seperti "ulasan" secara langsung di bagian dashboard,
                    # karena dataset pengguna dapat memakai huruf besar/kecil atau nama
                    # lain. Ini mencegah error KeyError: 'Ulasan' / 'ulasan'.
                    result = result.rename(columns={text_col: "Ulasan", label_col: "Sentimen Aktual"})
                    result["Ulasan"] = result["Ulasan"].fillna("").astype(str)
                    result["Sentimen Aktual"] = result["Sentimen Aktual"].astype(str).str.strip()
                    result = result[(result["Ulasan"].str.strip() != "") & (result["Sentimen Aktual"] != "")]
                    if result.empty:
                        raise ValueError("Tidak ada baris yang memiliki ulasan dan label setelah pembersihan.")

                    if result["Sentimen Aktual"].nunique() < 2:
                        raise ValueError("Model Naive Bayes membutuhkan minimal dua kelas sentimen yang berbeda.")

                    with st.spinner("Preprocessing, TF-IDF, training Naive Bayes, dan evaluasi model..."):
                        result["Teks Preprocessing"] = preprocess_series(result["Ulasan"])
                        training = train_model(
                            result["Teks Preprocessing"],
                            result["Sentimen Aktual"],
                            test_size=0.2,
                            random_state=42,
                        )
                        result["Sentimen Prediksi"] = training["model"].predict(
                            training["vectorizer"].transform(result["Teks Preprocessing"])
                        )
                        result["Aspek"] = result["Ulasan"].apply(determine_aspects)

                        if date_col != "(Tidak ada)":
                            result["Tanggal"] = df.loc[result.index, date_col].values

                        metrics = evaluate_model(training["y_test"], training["y_pred"])

                        # Save the trained artifacts only after successful training.
                        models_dir = __import__("pathlib").Path("models")
                        models_dir.mkdir(parents=True, exist_ok=True)
                        joblib.dump(training["vectorizer"], models_dir / "tfidf.pkl")
                        joblib.dump(training["model"], models_dir / "naive_bayes.pkl")

                        st.session_state.dataset = result
                        st.session_state.models = training
                        st.session_state.metrics = metrics
                        st.session_state.test_data = {
                            "y_test": training["y_test"],
                            "y_pred": training["y_pred"],
                            "labels": training["labels"],
                        }
                        st.session_state.uploaded_name = uploaded.name
                        st.session_state.analysis_ready = True

                    st.success("Analisis berhasil. Dashboard siap digunakan.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Proses analisis gagal: {exc}")
        except Exception as exc:
            st.error(f"Gagal membaca CSV: {exc}")

    st.stop()


# =========================
# DATA READY
# =========================
df = st.session_state.dataset.copy()

# Pastikan nama kolom internal konsisten meskipun session_state berasal dari
# versi aplikasi lama atau pemetaan dataset memakai huruf besar/kecil berbeda.
def ensure_canonical_columns(frame):
    frame = frame.copy()
    normalized = {str(c).strip().casefold(): c for c in frame.columns}

    aliases = {
        "Ulasan": ["ulasan", "review", "komentar", "comment", "text", "teks", "feedback"],
        "Sentimen Aktual": ["sentimen aktual", "sentimen", "sentiment", "label", "kategori", "class", "kelas"],
        "Sentimen Prediksi": ["sentimen prediksi", "prediksi sentimen", "prediction", "predicted_sentiment"],
        "Teks Preprocessing": ["teks preprocessing", "preprocessing", "teks_preprocessing"],
        "Aspek": ["aspek", "aspect"],
        "Tanggal": ["tanggal", "tgl", "date", "waktu", "time", "datetime", "timestamp"],
    }

    for canonical, candidates in aliases.items():
        if canonical in frame.columns:
            continue
        source = None
        for candidate in candidates:
            key = candidate.casefold()
            if key in normalized:
                source = normalized[key]
                break
        if source is not None:
            frame[canonical] = frame[source]

    return frame


df = ensure_canonical_columns(df)

# Jika aplikasi dijalankan ulang dengan session_state dari versi lama dan
# kolom prediksi belum ada, jangan biarkan dashboard crash.
if "Ulasan" not in df.columns:
    st.error(
        "Kolom ulasan hasil analisis tidak ditemukan. Silakan kembali ke awal "
        "dan upload dataset lalu pilih kolom ulasan dengan benar."
    )
    st.stop()
if "Sentimen Prediksi" not in df.columns:
    st.error(
        "Kolom Sentimen Prediksi tidak ditemukan. Silakan jalankan analisis "
        "kembali setelah upload dataset."
    )
    st.stop()

models = st.session_state.models
metrics = st.session_state.metrics

st.markdown(f'<div class="nav-note">📁 Dataset aktif: <b>{st.session_state.uploaded_name}</b> · {len(df):,} ulasan · {len(df.columns):,} kolom hasil analisis</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Dashboard", "✏️ Input Ulasan", "🗃️ Data Ulasan",
    "📊 Grafik Sentimen", "🔎 Analisis Aspek", "📈 Metrik Performa"
])


# =========================
# DASHBOARD
# =========================
with tabs[0]:
    counts = sentiment_counts(df["Sentimen Prediksi"])
    total = len(df)

    summary_col, input_col, recent_col, chart_col = st.columns([1.05, 1.15, 1.15, 1.05], gap="medium")

    with summary_col:
        st.markdown('<div class="dash-card dark-card"><h3>Ringkasan Dashboard</h3>', unsafe_allow_html=True)
        for name, icon, cls in [("Positif", "👍", "pos"), ("Negatif", "👎", "neg"), ("Netral", "●", "neu")]:
            count = int(counts.get(name, 0))
            pct = count / total if total else 0
            if count > 0 or name in counts.index:
                html = f'<div class="summary-item"><div class="summary-icon {cls}">{icon}</div><div><div class="summary-label">Sentimen {name}:</div><div class="summary-value">{count:,} <span>({pct:.1%})</span></div></div></div>'
                st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with input_col:
        st.markdown('<div class="dash-card"><h3>Input Ulasan untuk Prediksi</h3>', unsafe_allow_html=True)
        input_text = st.text_area("Masukkan Ulasan Anda:", placeholder="Proses pembayaran sudah mudah tetapi hasil studi belum muncul.", height=115, key="dashboard_input")
        if st.button("Prediksi Sentimen", type="primary", width='stretch', key="dashboard_predict"):
            if not input_text.strip():
                st.warning("Ulasan tidak boleh kosong.")
            else:
                try:
                    pred, proba = predict_text(input_text, models["vectorizer"], models["model"])
                    display = label_display(pred)
                    css = {"Positif": "result-positive", "Negatif": "result-negative", "Netral": "result-neutral"}.get(display, "result-neutral")
                    confidence = max(proba.values()) if proba else None
                    st.markdown(f'<div class="{css}">Prediksi: {display}</div>', unsafe_allow_html=True)
                    if confidence is not None:
                        st.caption(f"Confidence: {confidence:.2%}")
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    with recent_col:
        st.markdown('<div class="dash-card"><h3>Data Ulasan Terbaru</h3>', unsafe_allow_html=True)
        recent_cols = ["Ulasan"]
        if "Tanggal" in df.columns:
            recent_cols.append("Tanggal")
        recent_cols.append("Sentimen Prediksi")
        recent = df[recent_cols].tail(6).copy()
        recent["Sentimen Prediksi"] = recent["Sentimen Prediksi"].map(label_display)
        st.dataframe(recent, width='stretch', hide_index=True, height=245)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col:
        st.markdown('<div class="dash-card"><h3>Grafik Sentimen</h3>', unsafe_allow_html=True)
        chart_df = counts.rename_axis("Sentimen").reset_index(name="Jumlah")
        cbar, cpie = st.columns(2)
        with cbar:
            fig = px.bar(chart_df, x="Sentimen", y="Jumlah", text="Jumlah")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=245, margin=dict(l=0,r=0,t=10,b=45), showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        with cpie:
            fig = px.pie(chart_df, names="Sentimen", values="Jumlah", hole=.48)
            fig.update_traces(textinfo="percent")
            fig.update_layout(height=245, margin=dict(l=0,r=0,t=10,b=10), showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    aspect_col, metric_col, cm_col = st.columns([2.05, 1.0, 1.0], gap="medium")
    ar = aspect_rows(df)

    with aspect_col:
        st.markdown('<div class="dash-card"><h3>Analisis Berdasarkan Aspek Layanan Akademik</h3>', unsafe_allow_html=True)
        if ar.empty:
            st.info("Belum ada ulasan yang terdeteksi mengandung keyword aspek layanan akademik.")
        else:
            aspect_counts = pd.crosstab(ar["Aspek"], ar["Sentimen"]).reindex(ASPECTS, fill_value=0)
            for col in ["Positif", "Negatif", "Netral"]:
                if col not in aspect_counts.columns:
                    aspect_counts[col] = 0
            aspect_counts = aspect_counts[["Positif", "Negatif", "Netral"]].reset_index()
            fig = px.bar(aspect_counts.melt(id_vars="Aspek", var_name="Sentimen", value_name="Jumlah"), x="Aspek", y="Jumlah", color="Sentimen", barmode="group", text="Jumlah")
            fig.update_traces(textposition="inside")
            fig.update_layout(height=410, margin=dict(l=5,r=5,t=10,b=85), xaxis_tickangle=-25, legend=dict(orientation="h", y=1.06, x=0))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with metric_col:
        st.markdown('<div class="dash-card"><h3>Metrik Performa Model</h3>', unsafe_allow_html=True)
        for title, value in [("Akurasi", metrics["accuracy"]), ("Presisi", metrics["precision"]), ("Recall", metrics["recall"]), ("F1-Score", metrics["f1"])]:
            st.markdown(f'<div class="mini-metric"><div class="mini-value">{value:.1%}</div><div>{title}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cm_col:
        st.markdown('<div class="dash-card"><h3>Confusion Matrix</h3>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        y_test = st.session_state.test_data["y_test"]
        y_pred = st.session_state.test_data["y_pred"]
        labels = st.session_state.test_data["labels"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[label_display(x) for x in labels], columns=[label_display(x) for x in labels])
        fig = px.imshow(cm_df, text_auto=True, aspect="auto", labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=5,b=35), coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# INPUT ULASAN
# =========================
with tabs[1]:
    st.subheader("Input Ulasan untuk Prediksi")
    text = st.text_area(
        "Masukkan Ulasan Anda",
        placeholder="Proses pembayaran sudah mudah tetapi hasil studi belum muncul.",
        height=180,
        key="input_tab_text",
    )
    if st.button("Prediksi Sentimen", type="primary", key="input_tab_predict"):
        if not text.strip():
            st.warning("Ulasan tidak boleh kosong.")
        else:
            try:
                pred, proba = predict_text(text, models["vectorizer"], models["model"])
                display = label_display(pred)
                st.success(f"Sentimen: **{display}**")
                if proba:
                    st.write("**Probabilitas setiap kelas:**")
                    prob_df = pd.DataFrame(
                        {"Sentimen": [label_display(k) for k in proba], "Probability": list(proba.values())}
                    )
                    prob_df["Probability"] = prob_df["Probability"].map(lambda x: f"{x:.2%}")
                    st.dataframe(prob_df, hide_index=True, width='stretch')
            except Exception as exc:
                st.error(f"Prediksi gagal: {exc}")


# =========================
# DATA ULASAN
# =========================
with tabs[2]:
    st.subheader("Data Ulasan")
    c1, c2 = st.columns([1, 2])
    filter_sentiment = c1.selectbox("Filter Sentimen", ["Semua", "Positif", "Negatif", "Netral"])
    search = c2.text_input("Cari berdasarkan teks ulasan", placeholder="Ketik kata pencarian...")

    view = df.copy()
    view["Sentimen Prediksi"] = view["Sentimen Prediksi"].map(label_display)
    if filter_sentiment != "Semua":
        view = view[view["Sentimen Prediksi"] == filter_sentiment]
    if search.strip():
        mask = view["Ulasan"].astype(str).str.contains(search.strip(), case=False, na=False)
        view = view[mask]

    output_cols = ["Ulasan", "Teks Preprocessing"]
    if "Sentimen Aktual" in view.columns:
        output_cols.append("Sentimen Aktual")
    output_cols.append("Sentimen Prediksi")
    if "Aspek" in view.columns:
        output_cols.append("Aspek")
    if "Tanggal" in view.columns:
        output_cols.append("Tanggal")

    display_df = view[output_cols].copy()
    display_df.insert(0, "No", range(1, len(display_df) + 1))
    display_df["Aspek"] = display_df["Aspek"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Defensive duplicate-column protection.
    display_df.columns = make_unique_columns(display_df.columns)
    st.caption(f"Menampilkan {len(display_df):,} dari {len(df):,} ulasan.")
    st.dataframe(display_df, width='stretch', height=560, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Hasil Analisis CSV",
        data=csv_bytes,
        file_name="hasil_analisis_sentimen.csv",
        mime="text/csv",
        width='content',
    )


# =========================
# GRAFIK SENTIMEN
# =========================
with tabs[3]:
    st.subheader("Grafik Sentimen")
    counts = sentiment_counts(df["Sentimen Prediksi"])
    chart_df = counts.rename_axis("Sentimen").reset_index(name="Jumlah")
    chart_df["Persentase"] = chart_df["Jumlah"] / chart_df["Jumlah"].sum()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(chart_df, x="Sentimen", y="Jumlah", text="Jumlah")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Jumlah Ulasan", xaxis_title="")
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.pie(chart_df, names="Sentimen", values="Jumlah", hole=.48)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, width='stretch')

    st.dataframe(
        chart_df.assign(Persentase=chart_df["Persentase"].map(lambda x: f"{x:.2%}")),
        width='stretch', hide_index=True
    )


# =========================
# ANALISIS ASPEK
# =========================
with tabs[4]:
    st.subheader("Analisis Aspek Layanan Akademik")
    st.caption("Aspek ditentukan menggunakan rule-based keyword matching dan satu ulasan dapat memiliki lebih dari satu aspek.")

    ar = aspect_rows(df)
    if ar.empty:
        st.warning("Tidak ada aspek yang terdeteksi dari keyword yang ditentukan.")
    else:
        table = pd.crosstab(ar["Aspek"], ar["Sentimen"]).reindex(ASPECTS, fill_value=0)
        for col in ["Positif", "Negatif", "Netral"]:
            if col not in table.columns:
                table[col] = 0
        table = table[["Positif", "Negatif", "Netral"]]
        table["Total"] = table.sum(axis=1)
        table = table.reset_index()

        st.dataframe(table, width='stretch', hide_index=True)

        fig = px.bar(
            table.melt(id_vars="Aspek", value_vars=["Positif", "Negatif", "Netral"],
                       var_name="Sentimen", value_name="Jumlah"),
            x="Aspek", y="Jumlah", color="Sentimen", barmode="group", text="Jumlah"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=500, xaxis_tickangle=-35)
        st.plotly_chart(fig, width='stretch')

        neg_row = table.loc[table["Negatif"].idxmax()]
        pos_row = table.loc[table["Positif"].idxmax()]
        a, b = st.columns(2)
        with a:
            st.metric("Aspek dengan Sentimen Negatif Terbanyak", neg_row["Aspek"], f'{int(neg_row["Negatif"]):,} ulasan')
        with b:
            st.metric("Aspek dengan Sentimen Positif Terbanyak", pos_row["Aspek"], f'{int(pos_row["Positif"]):,} ulasan')

    st.subheader("Top Kata Berdasarkan TF-IDF")
    top_words = get_top_tfidf_words(
        models["vectorizer"], models["X_all"], n=20
    )
    if top_words:
        word_df = pd.DataFrame(top_words, columns=["Kata", "Skor TF-IDF"])
        fig = px.bar(word_df.sort_values("Skor TF-IDF"), x="Skor TF-IDF", y="Kata", orientation="h", text="Skor TF-IDF")
        fig.update_layout(height=560)
        st.plotly_chart(fig, width='stretch')


# =========================
# METRIK PERFORMA
# =========================
with tabs[5]:
    st.subheader("Metrik Performa Model (Naive Bayes + TF-IDF)")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, value in [
        (c1, "Accuracy", metrics["accuracy"]),
        (c2, "Precision", metrics["precision"]),
        (c3, "Recall", metrics["recall"]),
        (c4, "F1-Score", metrics["f1"]),
    ]:
        col.metric(title, format_pct(value))

    st.write("")
    y_test = st.session_state.test_data["y_test"]
    y_pred = st.session_state.test_data["y_pred"]
    labels = st.session_state.test_data["labels"]

    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[label_display(x) for x in labels],
        columns=[label_display(x) for x in labels],
    )
    fig = px.imshow(
        cm_df, text_auto=True, aspect="auto",
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
    )
    fig.update_layout(height=460)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Detail Evaluasi")
    report = metrics["report"].copy()
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df, width='stretch')

    st.caption("Metrik dihitung pada data uji (20%) yang dipisahkan saat proses training, dengan random_state=42.")


# =========================
# SIDEBAR / RESET
# =========================
with st.sidebar:
    st.header("Pengaturan")
    st.write("Dataset aktif:")
    st.code(st.session_state.uploaded_name or "-")
    if st.button("🔄 Reset & Upload Dataset Lain"):
        for key in ["dataset", "analysis_ready", "uploaded_name", "models", "metrics", "test_data"]:
            st.session_state[key] = None
        st.rerun()

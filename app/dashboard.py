import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "test_sample.parquet"

st.set_page_config(page_title="SOC Alert Triage", layout="wide")
st.markdown(
    "<style>[data-testid='stMainBlockContainer']{direction:rtl;text-align:right}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all():
    model = keras.models.load_model(MODELS / "classifier_weighted.keras")
    thr = float(np.load(MODELS / "clf_threshold.npy"))
    mean = np.load(MODELS / "norm_mean.npy")
    std = np.maximum(np.sqrt(np.load(MODELS / "norm_var.npy")), 1e-7)
    features = pd.read_csv(MODELS / "features.csv").iloc[:, 0].tolist()
    bg = np.load(MODELS / "shap_background.npy")

    def f(X):
        return model.predict(X, batch_size=4096, verbose=0).ravel()

    explainer = shap.Explainer(f, bg, feature_names=features)
    return thr, mean, std, features, explainer, f


@st.cache_data
def load_data():
    return pd.read_parquet(DATA)


def severity(p, thr):

    if p <= thr:
        return "طبيعي"
    if p >= 0.99:
        return "حرجة"
    if p >= 0.95:
        return "عالية"
    return "متوسطة"


thr, mean, std, features, explainer, f = load_all()
data = load_data()
X = data[features].to_numpy("float32")
labels = data["Label"].to_numpy()
N = len(data)

ss = st.session_state
ss.setdefault("i", 0)
ss.setdefault("log", {})
ss.setdefault("shap", {})


st.sidebar.header("التحكم")
if st.sidebar.button("السجل التالي"):
    ss.i += 1
play = st.sidebar.toggle("تشغيل تلقائي")
speed = st.sidebar.slider("ثواني بين السجلات", 0.5, 5.0, 1.5, 0.5)
if st.sidebar.button("إعادة من البداية"):
    ss.i, ss.log, ss.shap = 0, {}, {}
st.sidebar.caption(f"عتبة القرار: {thr:.4f} (FPR 0.5% على بيانات التحقق)")


i = ss.i % N
x_raw = X[i]
x_n = np.clip((x_raw - mean) / std, -10, 10).astype("float32")
p = float(f(x_n[None, :])[0])
is_attack = p > thr
ss.log[i] = (labels[i], p, is_attack)

st.title("لوحة فرز تنبيهات الـ SOC")
st.caption(f"السجل {i + 1} من {N} | عيّنة عرض من الـ Test (مو لقياس الأداء)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("القرار", "هجوم" if is_attack else "طبيعي")
c2.metric("احتمال الهجوم", f"{p:.3f}")
c3.metric("الخطورة", severity(p, thr))
truth = labels[i]
ok = (truth != "BENIGN") == is_attack
c4.metric("الحقيقة (للعرض فقط)", truth, "صح" if ok else "خطأ",
          delta_color="normal" if ok else "inverse")


if i not in ss.shap:
    with st.spinner("حساب SHAP..."):
        ss.shap[i] = explainer(x_n[None, :]).values[0]
sv = ss.shap[i]
top = np.argsort(-np.abs(sv))[:8]
contrib = pd.DataFrame({
    "الخاصية": [features[j] for j in top],
    "القيمة": [float(x_raw[j]) for j in top],
    "SHAP": [float(sv[j]) for j in top],
})
st.subheader("لماذا هذا القرار؟ (أهم 8 خصائص)")
st.caption("SHAP موجب = يدفع نحو هجوم، سالب = يدفع نحو طبيعي. تفسير لقرار الموديل، مو إثبات للسبب.")
left, right = st.columns(2)
left.dataframe(contrib, hide_index=True, use_container_width=True)
right.bar_chart(contrib.set_index("الخاصية")["SHAP"])

# ---------------- log ----------------
log = ss.log
alerts = sum(1 for v in log.values() if v[2])
st.subheader("سجل التنبيهات")
st.write(f"سجلات معالجة: {len(log)} | إنذارات: {alerts}")
recent = list(log.items())[-10:][::-1]
st.dataframe(
    pd.DataFrame({
        "السجل": [k + 1 for k, _ in recent],
        "الحقيقة": [v[0] for _, v in recent],
        "الاحتمال": [round(v[1], 3) for _, v in recent],
        "القرار": ["هجوم" if v[2] else "طبيعي" for _, v in recent],
    }),
    hide_index=True,
    use_container_width=True,
)

if play:
    time.sleep(speed)
    ss.i += 1
    st.rerun()

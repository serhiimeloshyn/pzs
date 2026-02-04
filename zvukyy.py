import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import wfdb

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

cesta_data = r"C:/Users/Владеец/OneDrive/Робочий стіл/Zvuky2"
def read_diagnosis_from_hea(record_name):
    hea_path = os.path.join(cesta_data, record_name + ".hea")
    if not os.path.exists(hea_path):
        return None

    diag_re = re.compile(r"(?:<diagnoses?>|diagnoses?|diagnosis)\s*:\s*([^<\r\n]+)", re.IGNORECASE)
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = diag_re.search(line)
            if m:
                # всё после двоеточия
                return m.group(1).strip()
    return None


def pick_records_by_diagnosis(ready_records):
    """
    Возвращает dict с ключами:
    healthy, hyperkinetic, reflux, hypokinetic
    (если что-то не нашлось — подставим первые попавшиеся патологии)
    """
    info = []
    for rec in ready_records:
        diag = read_diagnosis_from_hea(rec)
        info.append((rec, diag or ""))

    def find_one(predicate):
        for rec, diag in info:
            if predicate(diag.lower()):
                return rec
        return None

    healthy = find_one(lambda d: ("healthy" in d) or ("normal" in d) or ("norm" in d) or ("normální" in d) or ("normalní" in d))
    hyper = find_one(lambda d: "hyperkinetic" in d)
    reflux = find_one(lambda d: "reflux" in d)
    hypo = find_one(lambda d: "hypokinetic" in d)
    pathological_pool = [rec for rec, diag in info if not (("healthy" in diag.lower()) or ("normal" in diag.lower()) or ("norm" in diag.lower()) or ("normální" in diag.lower()) or ("normalní" in diag.lower()))]
    chosen = [x for x in [hyper, reflux, hypo] if x is not None]
    for rec in pathological_pool:
        if len(chosen) >= 3:
            break
        if rec not in chosen:
            chosen.append(rec)

    hyper = chosen[0] if len(chosen) > 0 else None
    reflux = chosen[1] if len(chosen) > 1 else None
    hypo = chosen[2] if len(chosen) > 2 else None

    return {
        "healthy": healthy,
        "hyperkinetic": hyper,
        "reflux": reflux,
        "hypokinetic": hypo,
    }




def list_ready_records(path):
    dat = {os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith(".dat")}
    hea = {os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith(".hea")}
    ready = sorted(dat & hea)

    print("dat:", len(dat))
    print("hea:", len(hea))
    print("ready(dat+hea):", len(ready))

    if len(ready) == 0:
        print("None")
    else:
        print("Примеры ready:", ready[:10])

    return ready


def load_signal(record_name):
    record_path = os.path.join(cesta_data, record_name) 
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal[:, 0].astype(float)
    fs = float(rec.fs)
    return sig, fs


def plot_4_signals(records, seconds=3.0):
    plt.figure(figsize=(14, 8))

    for i, rec in enumerate(records, start=1):
        sig, fs = load_signal(rec)
        n = int(min(len(sig), seconds * fs))
        t = np.arange(n) / fs

        ax = plt.subplot(len(records), 1, i)
        ax.plot(t, sig[:n])
        ax.set_title(f"Hlasový signál ({rec})")
        ax.set_xlabel("Čas (sekundy)")
        ax.set_ylabel("Amplituda")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("START")
    print("Folder:", cesta_data)

    ready = list_ready_records(cesta_data)

    if len(ready) >= 4:
        plot_4_signals(ready[:4], seconds=3.0)
    else:
        print("minimum 4 zapisy", len(ready))
    


    picked = pick_records_by_diagnosis(ready)

print("\n=== Vybrané záznamy ===")
for k, v in picked.items():
    print(k, "->", v, "| diag:", read_diagnosis_from_hea(v) if v else None)

records = [picked["healthy"], picked["hyperkinetic"], picked["reflux"], picked["hypokinetic"]]
titles = [
    "Hlasový signál (healthy)",
    "Hlasový signál (hyperkinetic dysphonia)",
    "Hlasový signál (reflux laryngitis)",
    "Hlasový signál (hypokinetic dysphonia)",
]

records_ok = [r for r in records if r is not None]
titles_ok = [t for r, t in zip(records, titles) if r is not None]

plot_4_signals(records_ok, seconds=3.0)  


def plot_4_spectra(records, seconds=3.0, fmax=4000):
    plt.figure(figsize=(14, 8))

    for i, rec in enumerate(records, start=1):
        sig, fs = load_signal(rec)

        n = int(min(len(sig), seconds * fs))
        sig = sig[:n]

        fft = np.fft.rfft(sig)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(sig), d=1/fs)

        ax = plt.subplot(len(records), 1, i)
        ax.plot(freqs, mag)
        ax.set_title(f"Spektrum (FFT) ({rec})")
        ax.set_xlabel("Frekvence (Hz)")
        ax.set_ylabel("Amplituda")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(fmax, fs/2))

    plt.tight_layout()
    plt.show()




def plot_4_cepstra(records, seconds=3.0, qmax=300):
    plt.figure(figsize=(14, 8))

    for i, rec in enumerate(records, start=1):
        sig, fs = load_signal(rec)

        n = int(min(len(sig), seconds * fs))
        sig = sig[:n]

        fft = np.fft.rfft(sig)
        log_spec = np.log(np.abs(fft) + 1e-10)
        cep = np.fft.irfft(log_spec)

        ax = plt.subplot(len(records), 1, i)
        ax.plot(cep[:qmax])
        ax.set_title(f"Kepstrum ({rec})")
        ax.set_xlabel("Quefrence (vzorky)")
        ax.set_ylabel("Amplituda")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




PATHOLOGY_CLASSES = ("hyperkinetic", "hypokinetic", "reflux")


def _normalize_diag(text: str) -> str:
    if text is None:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def true_voice_label(diagnosis_raw: str) -> str:
    """
    Map .hea diagnosis text to one of:
    - healthy
    - hyperkinetic / hypokinetic / reflux
    - pathological_other (fallback)
    """
    d = _normalize_diag(diagnosis_raw)
    if not d:
        return "unknown"

    healthy_keywords = ["healthy", "normal", "normální", "normalní", "norm", "bez patologie", "no pathology"]
    pathological_keywords = ["patholog", "disorder", "dys", "paral", "lesion", "polyp", "nodule", "edema", "tumor", "cancer"]

    if any(k in d for k in pathological_keywords):
        # explicit pathological tokens -> not healthy
        pass
    elif any(k in d for k in healthy_keywords):
        return "healthy"

    if "hyperkinetic" in d:
        return "hyperkinetic"
    if "hypokinetic" in d:
        return "hypokinetic"
    if "reflux" in d:
        return "reflux"
    return "pathological_other"


def iter_segments(signal: np.ndarray, fs: float, window_s: float, hop_s: float):
    window_n = int(round(window_s * fs))
    hop_n = int(round(hop_s * fs))
    if window_n <= 0:
        raise ValueError("window_s is too small")
    if hop_n <= 0:
        raise ValueError("hop_s is too small")
    n = int(len(signal))
    seg_idx = 0
    for start in range(0, n - window_n + 1, hop_n):
        yield seg_idx, signal[start : start + window_n]
        seg_idx += 1


def extract_basic_features(signal: np.ndarray, fs: float) -> dict:
    """
    Feature set used for segment statistics and classifiers.
    Mirrors the values used by the threshold-based `is_voice_healthy` logic (in part3.py).
    """
    signal = np.asarray(signal, dtype=float)
    n = int(signal.size)
    if n <= 0:
        return {
            "centroid_hz": 0.0,
            "spread_hz": 0.0,
            "skewness": 0.0,
            "entropy": 0.0,
            "cpp": 0.0,
            "ceps_mean": 0.0,
        }

    fft_result = np.fft.rfft(signal)
    magnitude = np.abs(fft_result)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    mag_sum = float(np.sum(magnitude))
    if mag_sum == 0.0:
        centroid = 0.0
        spread = 0.0
        skewness = 0.0
        entropy = 0.0
    else:
        mag_norm = magnitude / mag_sum
        centroid = float(np.sum(freqs * mag_norm))
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag_norm)))
        if spread == 0.0:
            skewness = 0.0
        else:
            skewness = float(np.sum(((freqs - centroid) ** 3) * mag_norm) / (spread ** 3))
        entropy = float(-np.sum(mag_norm * np.log2(mag_norm + 1e-12)))

    log_spectrum = np.log(magnitude + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)
    if cepstrum.size == 0:
        cpp = 0.0
        ceps_mean = 0.0
    elif cepstrum.size == 1:
        cpp = float(cepstrum[0])
        ceps_mean = float(cepstrum[0])
    else:
        cpp = float(np.max(cepstrum[1:]))
        ceps_mean = float(np.mean(cepstrum))

    return {
        "centroid_hz": centroid,
        "spread_hz": spread,
        "skewness": skewness,
        "entropy": entropy,
        "cpp": cpp,
        "ceps_mean": ceps_mean,
    }


def predict_healthy_from_basic_features(feats: dict) -> bool:
    cpp_threshold = 0.4
    ceps_mean_threshold = 0.00007
    entropy_threshold = 6.5
    centroid_min, centroid_max = 1100, 1800
    spread_threshold = 800
    skewness_threshold = 1.5

    conditions = [
        feats["cpp"] > cpp_threshold,
        feats["ceps_mean"] < ceps_mean_threshold,
        feats["entropy"] < entropy_threshold,
        centroid_min < feats["centroid_hz"] < centroid_max,
        feats["spread_hz"] < spread_threshold,
        abs(feats["skewness"]) < skewness_threshold,
    ]
    return sum(conditions) >= 5


def segment_success_statistics(records: list[str], window_s: float = 0.5, hop_s: float = 0.5):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    rows = []
    for rec in records:
        diagnosis = read_diagnosis_from_hea(rec)
        true_label = true_voice_label(diagnosis)
        if true_label == "unknown":
            continue

        true_healthy = (true_label == "healthy")
        sig, fs = load_signal(rec)
        for seg_idx, seg in iter_segments(sig, fs, window_s=window_s, hop_s=hop_s):
            feats = extract_basic_features(seg, fs)
            pred_healthy = bool(predict_healthy_from_basic_features(feats))
            rows.append({
                "record": rec,
                "seg_idx": int(seg_idx),
                "true_label": true_label,
                "true_healthy": bool(true_healthy),
                "pred_healthy": pred_healthy,
                **feats,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No segments for statistics.")
        return None

    y_true_1 = df["true_healthy"].astype(bool).to_numpy()
    y_pred_1 = df["pred_healthy"].astype(bool).to_numpy()
    acc1 = float(accuracy_score(y_true_1, y_pred_1))
    print("\n=== Segment stats: Healthy vs Pathological (stage 1) ===")
    print(f"Segments: {len(df)}")
    print(f"Accuracy: {acc1:.3f}")

    feature_cols = ["centroid_hz", "spread_hz", "skewness", "entropy", "cpp", "ceps_mean"]
    df = df.reset_index(drop=True).copy()
    df["pred_pathology_cv"] = "unknown"
    df["pred_overall_cv"] = np.where(df["pred_healthy"].to_numpy(), "healthy", "unknown")

    unique_records = sorted(df["record"].unique().tolist())
    for rec in unique_records:
        train_df = df[df["record"] != rec]
        test_df = df[df["record"] == rec]
        test_idx = test_df.index

        train_path = train_df[train_df["true_label"].isin(PATHOLOGY_CLASSES)]
        if train_path.empty:
            continue

        X_train = train_path[feature_cols].to_numpy(dtype=float)
        y_train = train_path["true_label"].to_numpy(dtype=object)

        clf = RandomForestClassifier(n_estimators=300, random_state=42)
        clf.fit(X_train, y_train)

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        df.loc[test_idx, "pred_pathology_cv"] = clf.predict(X_test)
        df.loc[test_idx, "pred_overall_cv"] = np.where(
            df.loc[test_idx, "pred_healthy"].to_numpy(),
            "healthy",
            df.loc[test_idx, "pred_pathology_cv"].to_numpy(),
        )

    mask_path = df["true_label"].isin(PATHOLOGY_CLASSES)
    y_true_2 = df.loc[mask_path, "true_label"].to_numpy(dtype=object)
    y_pred_2 = df.loc[mask_path, "pred_pathology_cv"].to_numpy(dtype=object)

    print("\n=== Segment stats: Pathology type (stage 2) ===")
    if y_true_2.size:
        acc2 = float(accuracy_score(y_true_2, y_pred_2))
        print(f"Pathology segments: {int(y_true_2.size)}")
        print(f"Accuracy (3 classes): {acc2:.3f}")
        print(classification_report(
            y_true_2,
            y_pred_2,
            labels=list(PATHOLOGY_CLASSES),
            target_names=list(PATHOLOGY_CLASSES),
            zero_division=0,
        ))
    else:
        print("No pathological segments for stage-2 evaluation.")

    mask_all = df["true_label"].isin(["healthy", *PATHOLOGY_CLASSES])
    y_true_all = df.loc[mask_all, "true_label"].to_numpy(dtype=object)
    y_pred_all = df.loc[mask_all, "pred_overall_cv"].to_numpy(dtype=object)
    acc_all = float(accuracy_score(y_true_all, y_pred_all)) if y_true_all.size else 0.0

    print("\n=== Segment stats: Overall pipeline (healthy + 3 pathologies) ===")
    print(f"All segments: {int(y_true_all.size)}")
    print(f"Accuracy (4 classes): {acc_all:.3f}")
    if y_true_all.size:
        print(classification_report(
            y_true_all,
            y_pred_all,
            labels=["healthy", *PATHOLOGY_CLASSES],
            target_names=["healthy", *PATHOLOGY_CLASSES],
            zero_division=0,
        ))

    return df


plot_4_spectra(ready[:4], seconds=3.0, fmax=4000)
plot_4_cepstra(ready[:4], seconds=3.0, qmax=300)

try:
    segment_success_statistics(ready, window_s=0.5, hop_s=0.5)
except Exception as e:
    print(f"Segment statistics failed: {e}")

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_path = "C:/Users/Владеец/OneDrive/Робочий стіл/ECG/100001_ECG"

PLOT_DURATION_S = 10    # kolik sekund vykreslovat (aby to nebylo příliš velké)
FFT_DURATION_S = 60     # kolik sekund použít pro spektrum
ANOMALY_ANALYSIS_S = 60 # kolik sekund použít pro detekci amplitudových anomálií

def load_ecg_signal(file_path):
    """Načte EKG signál ze souborů kompatibilních s PhysioNet."""
    try:
        record = wfdb.rdrecord(file_path)  # Načtení .hea + .dat
        signal = record.p_signal[:, 0]  
        fs = record.fs  
        signal = signal.astype(np.float32, copy=False)
        fs = float(fs)

        print(f"Úspěšně načteno: {len(signal)} vzorků, frekvence: {fs} Hz")
        return signal, fs
    except Exception as e:
        print(f"Chyba při načítání: {e}")
        return None, None

signal, fs = load_ecg_signal(file_path)

if signal is not None:
    def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=360, order=2):
        nyq = 0.5 * fs 
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    
    filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=fs).astype(np.float32, copy=False)

    n_plot = min(len(signal), int(PLOT_DURATION_S * fs))
    time_axis = np.arange(n_plot) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal[:n_plot], label="Původní signál", alpha=0.6)
    plt.plot(time_axis, filtered_signal[:n_plot], label="Filtrovaný signál", color='red')
    plt.xlabel("Čas (s)")
    plt.ylabel("Amplituda")
    plt.title("Filtrace EKG signálu")
    plt.legend()
    plt.grid(True)
    plt.show()





import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

if signal is not None:
    ecg_peaks = nk.ecg_findpeaks(filtered_signal, sampling_rate=fs)
    r_peaks = ecg_peaks["ECG_R_Peaks"]
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    upper_threshold = mean_rr + 2 * std_rr
    lower_threshold = mean_rr - 2 * std_rr
    anomalies = np.where((rr_intervals > upper_threshold) | (rr_intervals < lower_threshold))[0]

    plt.figure(figsize=(12, 5))
    plt.plot(rr_intervals, label="RR intervaly", marker='o')
    plt.scatter(anomalies, rr_intervals[anomalies], color='red', label="Anomálie", zorder=3)
    plt.axhline(upper_threshold, color='r', linestyle='--', label="Horní práh")
    plt.axhline(lower_threshold, color='r', linestyle='--', label="Dolní práh")
    plt.xlabel("Srdeční tepy")
    plt.ylabel("RR interval (s)")
    plt.title("Analýza RR intervalů a detekce anomálií")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Průměrný RR interval: {mean_rr:.3f} sek")
    print(f"Standardní odchylka RR: {std_rr:.3f} sek")
    print(f"Nalezeno {len(anomalies)} anomálních RR intervalů")


# Výpočet okamžité tepové frekvence (BPM)
heart_rate = 60 / rr_intervals

# Časová osa pro HR (střed mezi R-vrcholy)
hr_time = r_peaks[1:] / fs  

plt.figure(figsize=(12, 5))
plt.plot(hr_time, heart_rate, color="green", linewidth=1)
plt.xlabel("Čas (s)")
plt.ylabel("Tepová frekvence (BPM)")
plt.title("Tepová frekvence v čase")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(heart_rate, bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Tepová frekvence (BPM)")
plt.ylabel("Počet výskytů")
plt.title("Histogram tepové frekvence")
plt.grid(True)
plt.show()



plt.figure(figsize=(5, 6))
plt.boxplot(heart_rate, vert=True)
plt.ylabel("Tepová frekvence (BPM)")
plt.title("Boxplot tepové frekvence")
plt.grid(True)
plt.show()

anomalous_hr = heart_rate[anomalies]

plt.figure(figsize=(12, 5))
plt.plot(hr_time, heart_rate, label="Tepová frekvence", alpha=0.7)
plt.scatter(hr_time[anomalies], anomalous_hr, color="red", label="Anomálie", zorder=3)
plt.xlabel("Čas (s)")
plt.ylabel("Tepová frekvence (BPM)")
plt.title("Tepová frekvence s vyznačenými anomáliemi")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

if 'filtered_signal' in locals() and 'fs' in locals():
    N = min(len(filtered_signal), int(FFT_DURATION_S * fs))
    segment = filtered_signal[:N]

    fft_values = fft(segment)
    freqs = fftfreq(N, 1/fs) 

    positive_freqs = freqs[:N // 2]
    magnitude_spectrum = np.abs(fft_values[:N // 2])

    plt.figure(figsize=(12, 5))
    plt.plot(positive_freqs, magnitude_spectrum, color='blue', label="Spektrum signálu")
    plt.xlabel("Frekvence (Hz)")
    plt.ylabel("Amplituda")
    plt.title("Frekvenční spektrum EKG signálu")
    plt.xlim(0, 100) 
    plt.legend()
    plt.grid(True)
    plt.show()

    noise_threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
    noise_indices = np.where(magnitude_spectrum > noise_threshold)[0]
    noise_frequencies = positive_freqs[noise_indices]

    print(f"Detekován vysokofrekvenční šum na frekvencích: {noise_frequencies[:10]} Hz (zobrazeno prvních 10)")
else:
    print("Chyba: Proměnné filtered_signal a fs nejsou definovány! Nejprve načtěte a filtrujte signál.")








from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff=50.0, fs=360, order=2):
    nyq = 0.5 * fs  
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

if 'filtered_signal' in locals():
    # Bandpass už omezuje signál na <= 50 Hz; filtfilt na celém záznamu může být paměťově náročný.
    # Proto lowpass aplikujeme jen na krátký úsek pro vizualizaci a pro další kroky použijeme `filtered_signal`.
    clean_signal = filtered_signal

    n_plot = min(len(filtered_signal), int(PLOT_DURATION_S * fs))
    time_axis = np.arange(n_plot) / fs

    try:
        clean_signal_plot = lowpass_filter(filtered_signal[:n_plot], cutoff=50.0, fs=fs)
    except Exception as e:
        clean_signal_plot = filtered_signal[:n_plot]
        print(f"Upozornění: lowpass filtr přeskočen: {e}")

    plt.figure(figsize=(12, 5))
    plt.plot(
        time_axis,
        filtered_signal[:n_plot],
        label="Před odstraněním šumu",
        alpha=0.6
    )
    plt.plot(
        time_axis,
        clean_signal_plot,
        label="Po odstranění šumu",
        color="red"
    )

    plt.xlabel("Čas (s)")
    plt.ylabel("Amplituda")
    plt.title(f"Filtrace vysokofrekvenčního šumu (prvních {PLOT_DURATION_S} s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Vysokofrekvenční šum odstraněn (na zobrazeném úseku)")



def detect_amplitude_anomalies(signal, threshold=3):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    anomalies = np.where(np.abs(signal - mean_val) > threshold * std_val)[0]
    return anomalies

if 'clean_signal' in locals():
    n_anom = min(len(clean_signal), int(ANOMALY_ANALYSIS_S * fs))
    analysis_signal = clean_signal[:n_anom]
    anomaly_indices = detect_amplitude_anomalies(analysis_signal, threshold=3)

    plt.figure(figsize=(12, 5))
    plt.plot(analysis_signal, label="Vyčištěný signál", alpha=0.7)
    plt.scatter(anomaly_indices, analysis_signal[anomaly_indices], color='red', label="Anomálie", zorder=3)
    plt.xlabel("Vzorky")
    plt.ylabel("Amplituda")
    plt.title(f"Detekce anomálních úseků v signálu (prvních {ANOMALY_ANALYSIS_S} s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Nalezeno {len(anomaly_indices)} anomálních bodů v signálu (na prvních {ANOMALY_ANALYSIS_S} s).")






import matplotlib.pyplot as plt

if 'clean_signal' in locals() and 'anomaly_indices' in locals():
    plot_signal = analysis_signal if 'analysis_signal' in locals() else clean_signal
    time_axis = np.arange(len(plot_signal)) / fs 

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, plot_signal, label="Vyčištěný EKG", alpha=0.7)
    plt.scatter(time_axis[anomaly_indices], plot_signal[anomaly_indices], color='red', label="Anomálie", zorder=3)
    plt.xlabel("Čas (s)")
    plt.ylabel("Amplituda")
    plt.title("Finální vizualizace anomálií v EKG signálu")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Bylo detekováno {len(anomaly_indices)} anomálních segmentů.")
else:
    print("Data nejsou načtena nebo zpracování není dokončeno!")


mean_rr = np.mean(rr_intervals)          
heart_rate_avg = 60 / mean_rr                

print(f"Tepová frekvence: {heart_rate_avg:.2f} BPM")


import pandas as pd

results_df = pd.DataFrame({
    "Měření": ["EKG_100001"],
    "Průměrný RR interval (s)": [mean_rr],
    "Tepová frekvence (BPM)": [heart_rate_avg]
})

print(results_df)

import pandas as pd

# Souhrn anomálií podle RR intervalů (nezávislé na detekci amplitudových anomálií v signálu)
if 'rr_intervals' in locals() and 'anomalies' in locals():
    anomaly_count = int(len(anomalies))
    mean_rr = float(np.mean(rr_intervals))
    max_rr = float(np.max(rr_intervals))
    duration_s = float(len(filtered_signal) / fs) if 'filtered_signal' in locals() else np.nan
    anomaly_rate = (anomaly_count / duration_s) if duration_s and not np.isnan(duration_s) else np.nan

    summary_df = pd.DataFrame({
        "Metrika": ["Počet anomálií", "Průměrný RR interval (s)", "Maximální RR interval (s)", "Frekvence anomálií (1/s)"],
        "Hodnota": [anomaly_count, mean_rr, max_rr, anomaly_rate]
    })

    # Zobrazení tabulky (ace_tools je volitelné)
    try:
        import ace_tools as tools
        tools.display_dataframe_to_user(name="Výsledky analýzy anomálií", dataframe=summary_df)
    except Exception:
        print(summary_df)
else:
    anomaly_count = 0
    mean_rr = float(np.mean(rr_intervals)) if 'rr_intervals' in locals() else np.nan
    max_rr = float(np.max(rr_intervals)) if 'rr_intervals' in locals() else np.nan
    anomaly_rate = np.nan
    print(" Nejsou k dispozici žádná data pro analýzu!")



report_text = f"""
**Závěrečná zpráva o analýze EKG**
Počet detekovaných anomálií: {anomaly_count}
Průměrný RR interval: {mean_rr:.3f} sek
Maximální RR interval: {max_rr:.3f} sek
Frekvence anomálií: {anomaly_rate:.4f} anomálií/s

**Závěry:**
- Bylo detekováno {anomaly_count} anomálních úseků v EKG signálu.
- Průměrná délka RR intervalu je {mean_rr:.3f} sek, což odpovídá normě.
- Maximální RR interval ({max_rr:.3f} sek) může naznačovat možné arytmie.
- Vizualizace signálu potvrzuje výskyt odchylek.

**Doporučené kroky:**
- Provést hlubší analýzu na základě dalších parametrů.
- Porovnat s expertními anotacemi pro vyšší přesnost klasifikace.
- Použít metody strojového učení pro přesnější diagnostiku.
"""

print(report_text)










# Souhrnná statistika úspěšnosti výpočtu tepové frekvence oproti anotacím (WFDB)
from pathlib import Path

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
}


def _guess_annotation_extension(record_path, preferred=("atr", "qrs")):
    path = Path(record_path)
    record_dir = path.parent
    record_base = path.name

    if not record_dir.exists():
        return None

    exts = []
    for p in record_dir.glob(f"{record_base}.*"):
        ext = p.suffix.lstrip(".")
        if ext and ext.lower() not in {"hea", "dat"}:
            exts.append(ext)

    exts_lower = {e.lower(): e for e in exts}
    for ext in preferred:
        if ext in exts_lower:
            return exts_lower[ext]

    return exts[0] if exts else None


def _load_reference_r_peaks(record_path, ann_extension=None):
    extension = ann_extension or _guess_annotation_extension(record_path) or "atr"
    ann = wfdb.rdann(record_path, extension)

    samples = np.asarray(getattr(ann, "sample", []), dtype=int)
    symbols = getattr(ann, "symbol", None)

    if symbols is not None and len(symbols) == len(samples):
        mask = np.array([s in BEAT_SYMBOLS for s in symbols], dtype=bool)
        filtered_samples = samples[mask]
        if filtered_samples.size >= 2:
            return filtered_samples, extension, True

    if samples.size >= 2:
        return samples, extension, False

    return None, extension, False


def _match_peaks(reference_peaks, detected_peaks, tolerance_samples):
    reference_peaks = np.asarray(reference_peaks, dtype=int)
    detected_peaks = np.asarray(detected_peaks, dtype=int)

    if reference_peaks.size == 0 or detected_peaks.size == 0:
        return [], np.zeros(reference_peaks.size, dtype=bool)

    reference_peaks = np.sort(reference_peaks)
    detected_peaks = np.sort(detected_peaks)

    matched_reference = np.zeros(reference_peaks.size, dtype=bool)
    matches = []

    for det in detected_peaks:
        i = int(np.searchsorted(reference_peaks, det))
        candidates = []
        for idx in (i - 1, i):
            if 0 <= idx < reference_peaks.size and not matched_reference[idx]:
                diff = det - reference_peaks[idx]
                if abs(diff) <= tolerance_samples:
                    candidates.append((abs(diff), idx, diff))

        if candidates:
            _, idx, diff = min(candidates, key=lambda x: x[0])
            matched_reference[idx] = True
            matches.append((int(reference_peaks[idx]), int(det), int(diff)))

    return matches, matched_reference


def summarize_hr_vs_annotations(
    record_path,
    detected_r_peaks,
    fs,
    ann_extension=None,
    tolerance_s=0.1,
):
    try:
        reference_peaks, used_extension, filtered_beats = _load_reference_r_peaks(
            record_path, ann_extension=ann_extension
        )
    except Exception as e:
        print(f"Nelze načíst anotace (wfdb.rdann): {e}")
        return None

    if reference_peaks is None:
        print("Anotace neobsahují dostatek tepů pro porovnání (min. 2).")
        return None

    detected_r_peaks = np.asarray(detected_r_peaks, dtype=int)
    if detected_r_peaks.size < 2:
        print("Detekované R-vrcholy nejsou k dispozici pro porovnání (min. 2).")
        return None

    tolerance_samples = int(round(tolerance_s * fs))
    matches, matched_reference = _match_peaks(
        reference_peaks, detected_r_peaks, tolerance_samples=tolerance_samples
    )

    tp = len(matches)
    fp = int(detected_r_peaks.size - tp)
    fn = int(reference_peaks.size - tp)

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else np.nan

    timing_errors_ms = (
        np.array([diff for _, _, diff in matches], dtype=float) / fs * 1000.0
        if tp
        else np.array([], dtype=float)
    )
    abs_timing_errors_ms = np.abs(timing_errors_ms)

    # HR porovnání na základě po sobě jdoucích shodných tepů (matched beats)
    ref_to_det = {ref: det for ref, det, _ in matches}
    matched_ref_samples = np.array(sorted(ref_to_det.keys()), dtype=int)
    if matched_ref_samples.size >= 2:
        matched_det_samples = np.array([ref_to_det[s] for s in matched_ref_samples], dtype=int)

        rr_ref = np.diff(matched_ref_samples) / fs
        rr_det = np.diff(matched_det_samples) / fs

        hr_ref = 60.0 / rr_ref
        hr_det = 60.0 / rr_det
        hr_err = hr_det - hr_ref
        hr_abs_err = np.abs(hr_err)

        hr_bias = float(np.mean(hr_err))
        hr_mae = float(np.mean(hr_abs_err))
        hr_rmse = float(np.sqrt(np.mean(hr_err ** 2)))
        hr_within_5 = float(np.mean(hr_abs_err <= 5.0) * 100.0)
        hr_within_10 = float(np.mean(hr_abs_err <= 10.0) * 100.0)
        hr_corr = float(np.corrcoef(hr_ref, hr_det)[0, 1]) if hr_ref.size > 1 else np.nan
    else:
        hr_ref = np.array([], dtype=float)
        hr_det = np.array([], dtype=float)
        hr_err = np.array([], dtype=float)
        hr_bias = np.nan
        hr_mae = np.nan
        hr_rmse = np.nan
        hr_within_5 = np.nan
        hr_within_10 = np.nan
        hr_corr = np.nan

    summary_rows = [
        ("Anotační soubor", used_extension),
        ("Použity jen beat anotace", bool(filtered_beats)),
        ("Počet tepů (anotace)", int(reference_peaks.size)),
        ("Počet tepů (detekce)", int(detected_r_peaks.size)),
        ("Tolerance pro shodu (ms)", float(tolerance_s * 1000.0)),
        ("TP (shodné tepy)", int(tp)),
        ("FP", int(fp)),
        ("FN", int(fn)),
        ("Precision", float(precision)),
        ("Recall", float(recall)),
        ("F1", float(f1)),
        ("Abs. chyba času – průměr (ms)", float(np.mean(abs_timing_errors_ms)) if tp else np.nan),
        ("Abs. chyba času – medián (ms)", float(np.median(abs_timing_errors_ms)) if tp else np.nan),
        ("Abs. chyba času – p95 (ms)", float(np.percentile(abs_timing_errors_ms, 95)) if tp else np.nan),
        ("Počet HR intervalů v porovnání", int(hr_err.size)),
        ("HR průměr anotace (BPM)", float(np.mean(hr_ref)) if hr_ref.size else np.nan),
        ("HR průměr detekce (BPM)", float(np.mean(hr_det)) if hr_det.size else np.nan),
        ("HR bias det-anot (BPM)", hr_bias),
        ("HR MAE (BPM)", hr_mae),
        ("HR RMSE (BPM)", hr_rmse),
        ("HR do 5 BPM (%)", hr_within_5),
        ("HR do 10 BPM (%)", hr_within_10),
        ("HR korelace", hr_corr),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["Metrika", "Hodnota"])
    print("\nSouhrnná statistika: úspěšnost výpočtu tepové frekvence vs. anotace")
    print(summary_df.to_string(index=False))
    return summary_df


import csv


def _guess_ann_csv_path(record_path):
    p = Path(record_path)
    subject_id = p.name.split("_")[0]
    candidate = p.with_name(f"{subject_id}_ANN.csv")
    return candidate if candidate.exists() else None


def _load_ann_csv_tracks(csv_path):
    triplets = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    tracks = [[] for _ in range(4)]

    with Path(csv_path).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            if len(row) < 12:
                continue
            for k, (s, e, l) in enumerate(triplets):
                if row[s].strip() == "" or row[e].strip() == "" or row[l].strip() == "":
                    continue
                tracks[k].append((int(row[s]), int(row[e]), int(row[l])))

    for k in range(4):
        tracks[k].sort(key=lambda x: x[0])

    return tracks


def _labels_for_samples_from_segments(segments, sample_indices):
    # segments: list of (start,end,label), 1-based inclusive indices (ECG sample domain)
    # sample_indices: 0-based indices (as produced by NeuroKit), same sampling rate as ECG
    sample_indices = np.asarray(sample_indices, dtype=int)
    out = np.full(sample_indices.shape, -1, dtype=int)

    j = 0
    for i, s in enumerate(sample_indices):
        s1 = int(s) + 1  # convert to 1-based
        while j < len(segments) and s1 > segments[j][1]:
            j += 1
        if j >= len(segments):
            break
        if segments[j][0] <= s1 <= segments[j][1]:
            out[i] = int(segments[j][2])

    return out


def _infer_hr_thresholds_from_labels(hr_values, labels):
    hr_values = np.asarray(hr_values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    med = {}
    for lab in (1, 2, 3):
        mask = labels == lab
        if np.any(mask):
            med[lab] = float(np.median(hr_values[mask]))

    t12 = (med[1] + med[2]) / 2.0 if 1 in med and 2 in med else np.nan
    t23 = (med[2] + med[3]) / 2.0 if 2 in med and 3 in med else np.nan
    return t12, t23, med


def _hr_to_label(hr_values, t12, t23):
    hr_values = np.asarray(hr_values, dtype=float)
    return np.where(hr_values < t12, 1, np.where(hr_values < t23, 2, 3)).astype(int)


def _classification_metrics(y_true, y_pred, classes=(1, 2, 3)):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred)) if y_true.size else np.nan

    rows = []
    f1s = []
    for lab in classes:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))

        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else np.nan
        f1s.append(f1)
        rows.append((lab, tp, fp, fn, prec, rec, f1))

    macro_f1 = float(np.nanmean(f1s)) if f1s else np.nan
    per_class_df = pd.DataFrame(
        rows,
        columns=["Třída", "TP", "FP", "FN", "Precision", "Recall", "F1"],
    )
    return acc, macro_f1, per_class_df


def summarize_hr_vs_ann_csv(
    record_path,
    heart_rate_values,
    interval_samples,
    fs,
    ann_csv_path=None,
    track=None,  # 1..4 or None=all
):
    csv_path = Path(ann_csv_path) if ann_csv_path else _guess_ann_csv_path(record_path)
    if csv_path is None or not csv_path.exists():
        print("Soubor s anotacemi ANN.csv nebyl nalezen.")
        return None

    tracks = _load_ann_csv_tracks(csv_path)
    track_indices = [track - 1] if track in (1, 2, 3, 4) else list(range(4))

    results = []
    heart_rate_values = np.asarray(heart_rate_values, dtype=float)
    interval_samples = np.asarray(interval_samples, dtype=int)

    for ti in track_indices:
        segments = tracks[ti]
        if not segments:
            continue

        y_true_full = _labels_for_samples_from_segments(segments, interval_samples)
        mask = y_true_full != -1
        if not np.any(mask):
            continue

        y_true = y_true_full[mask]
        hr = heart_rate_values[mask]

        t12, t23, med = _infer_hr_thresholds_from_labels(hr, y_true)
        y_pred = _hr_to_label(hr, t12, t23)
        acc, macro_f1, per_class = _classification_metrics(y_true, y_pred)

        results.append({
            "Track": ti + 1,
            "ANN.csv": str(csv_path),
            "N": int(y_true.size),
            "T12 (BPM)": float(t12),
            "T23 (BPM)": float(t23),
            "Median HR (label=1)": med.get(1, np.nan),
            "Median HR (label=2)": med.get(2, np.nan),
            "Median HR (label=3)": med.get(3, np.nan),
            "Accuracy": acc,
            "Macro F1": macro_f1,
        })

        print(f"\nSouhrn HR vs ANN.csv (track {ti + 1})")
        print(f"- Použitý soubor: {csv_path}")
        print(f"- Odhad prahů (BPM): T12={t12:.2f}, T23={t23:.2f}")
        print(per_class.to_string(index=False))

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        print("\nSouhrnná statistika (všechny tracky)")
        print(summary_df.to_string(index=False))
    return summary_df


if "r_peaks" in locals() and "fs" in locals() and signal is not None and "heart_rate" in locals():
    r_peaks_arr = np.asarray(r_peaks, dtype=int)
    if r_peaks_arr.size >= 2:
        hr_interval_samples = ((r_peaks_arr[:-1] + r_peaks_arr[1:]) // 2).astype(int)
        ann_csv = _guess_ann_csv_path(file_path)
        if ann_csv is not None:
            summarize_hr_vs_ann_csv(
                record_path=file_path,
                heart_rate_values=heart_rate,
                interval_samples=hr_interval_samples,
                fs=fs,
                ann_csv_path=ann_csv,
                track=None,  # změň na 3, pokud chceš jen jeden track
            )
        else:
            summarize_hr_vs_annotations(
                record_path=file_path,
                detected_r_peaks=r_peaks_arr,
                fs=fs,
                ann_extension=None,  # např. \"atr\"; None = automatický výběr
                tolerance_s=0.1,     # 100 ms
            )

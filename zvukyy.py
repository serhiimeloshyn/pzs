import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# 1) Укажи путь к папке с данными
cesta_data = r"C:/Users/Владеец/OneDrive/Робочий стіл/Zvuky2"


def read_diagnosis_from_hea(record_name):
    """
    Читает диагноз из voiceXXX.hea (ищет строку с Diagnosis:)
    Возвращает строку (может быть None, если не найдено)
    """
    hea_path = os.path.join(cesta_data, record_name + ".hea")
    if not os.path.exists(hea_path):
        return None

    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if "diagnosis" in low:
                # всё после двоеточия
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[-1].strip()
                return line.strip()
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

    # 1) healthy
    healthy = find_one(lambda d: ("healthy" in d) or ("normal" in d) or ("norm" in d) or ("normální" in d) or ("normalní" in d))

    # 2) конкретные патологии
    hyper = find_one(lambda d: "hyperkinetic" in d)
    reflux = find_one(lambda d: "reflux" in d)
    hypo = find_one(lambda d: "hypokinetic" in d)

    # список патологий "любые не-healthy"
    pathological_pool = [rec for rec, diag in info if not (("healthy" in diag.lower()) or ("normal" in diag.lower()) or ("norm" in diag.lower()) or ("normální" in diag.lower()) or ("normalní" in diag.lower()))]

    # добиваем пропуски любыми патологиями
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
        print("❗Нет ни одной пары .dat+.hea. WFDB не сможет читать записи.")
    else:
        print("Примеры ready:", ready[:10])

    return ready


def load_signal(record_name):
    record_path = os.path.join(cesta_data, record_name)  # без расширения
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
        print("Нужно минимум 4 записи с .dat+.hea. Сейчас есть:", len(ready))
    


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

# убираем None (если healthy не нашёлся и т.п.)
records_ok = [r for r in records if r is not None]
titles_ok = [t for r, t in zip(records, titles) if r is not None]

plot_4_signals(records_ok, seconds=3.0)  # или 5.0 как хочешь



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




plot_4_spectra(ready[:4], seconds=3.0, fmax=4000)
plot_4_cepstra(ready[:4], seconds=3.0, qmax=300)


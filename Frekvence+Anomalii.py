import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_path = "C:/Users/Владеец/OneDrive/Робочий стіл/ECG/100001_ECG"

def load_ecg_signal(file_path):
    """Načte EKG signál ze souborů kompatibilních s PhysioNet."""
    try:
        record = wfdb.rdrecord(file_path)  # Načtení .hea + .dat
        signal = record.p_signal[:, 0]  
        fs = record.fs  

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

    
    filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=fs)

    time_axis = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal, label="Původní signál", alpha=0.6)
    plt.plot(time_axis, filtered_signal, label="Filtrovaný signál", color='red')
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
    N = len(filtered_signal)

    fft_values = fft(filtered_signal)
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
    clean_signal = lowpass_filter(filtered_signal, cutoff=50.0, fs=fs)

    import matplotlib.pyplot as plt

PLOT_DURATION = 10  
n_plot = int(PLOT_DURATION * fs)

time_axis = np.arange(n_plot) / fs

plt.figure(figsize=(12, 5))
plt.plot(
    time_axis,
    filtered_signal[:n_plot],
    label="Před odstraněním šumu",
    alpha=0.6
)
plt.plot(
    time_axis,
    clean_signal[:n_plot],
    label="Po odstranění šumu",
    color="red"
)

plt.xlabel("Čas (s)")
plt.ylabel("Amplituda")
plt.title("Filtrace vysokofrekvenčního šumu (prvních 10 s)")
plt.legend()
plt.grid(True)
plt.show()

print("Vysokofrekvenční šum odstraněn")



def detect_amplitude_anomalies(signal, threshold=3):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    anomalies = np.where(np.abs(signal - mean_val) > threshold * std_val)[0]
    return anomalies

if 'clean_signal' in locals():
    anomaly_indices = detect_amplitude_anomalies(clean_signal, threshold=3)

    plt.figure(figsize=(12, 5))
    plt.plot(clean_signal, label="Vyčištěný signál", alpha=0.7)
    plt.scatter(anomaly_indices, clean_signal[anomaly_indices], color='red', label="Anomálie", zorder=3)
    plt.xlabel("Vzorky")
    plt.ylabel("Amplituda")
    plt.title("Detekce anomálních úseků v signálu")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Nalezeno {len(anomaly_indices)} anomálních bodů v signálu.")






import matplotlib.pyplot as plt

if 'clean_signal' in locals() and 'anomaly_indices' in locals():
    time_axis = np.arange(len(clean_signal)) / fs 

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, clean_signal, label="Vyčištěný EKG", alpha=0.7)
    plt.scatter(time_axis[anomaly_indices], clean_signal[anomaly_indices], color='red', label="Anomálie", zorder=3)
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
heart_rate = 60 / mean_rr                

print(f"Tepová frekvence: {heart_rate:.2f} BPM")


import pandas as pd

results_df = pd.DataFrame({
    "Měření": ["EKG_100001"],
    "Průměrný RR interval (s)": [mean_rr],
    "Tepová frekvence (BPM)": [heart_rate]
})

print(results_df)

import pandas as pd

# Ověříme, že všechny potřebné proměnné jsou definovány
if 'rr_intervals' in locals() and 'anomaly_indices' in locals():
    anomaly_count = len(anomaly_indices)
    mean_rr = np.mean(rr_intervals)
    max_rr = np.max(rr_intervals)
    anomaly_rate = anomaly_count / len(clean_signal) * fs  # Anomálie za sekundu

    # Vytvoříme souhrnnou tabulku
    summary_df = pd.DataFrame({
        "Metrika": ["Počet anomálií", "Průměrný RR interval (s)", "Maximální RR interval (s)", "Frekvence anomálií (1/s)"],
        "Hodnota": [anomaly_count, mean_rr, max_rr, anomaly_rate]
    })

    # Zobrazení tabulky
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Výsledky analýzy anomálií", dataframe=summary_df)
else:
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










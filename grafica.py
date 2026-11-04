# PERSONA 1 - Gráfica de los grupos (estados)
# Requiere haber corrido antes persona1_estados_medical.py (usa dataset_con_estados.csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_con_estados.csv")
nombres = df.groupby("estado")["estado_nombre"].first().to_dict()
colores = {0: "#2a9d8f", 1: "#e9c46a", 2: "#f4a261", 3: "#c0392b"}

# ---- Gráfica 1: dispersión de los pacientes (escala log), coloreados por estado ----
fig, ax = plt.subplots(figsize=(8, 6))
for e in sorted(nombres):
    d = df[df["estado"] == e]
    ax.scatter(d["log_Troponin"], d["log_CK-MB"], s=14, alpha=0.6,
               color=colores[e], label=f"{e} - {nombres[e]} (n={len(d)})")
    ax.scatter(d["log_Troponin"].mean(), d["log_CK-MB"].mean(), s=180, marker="X",
               color=colores[e], edgecolor="black", linewidth=1.2)  # centroide
ax.set_xlabel("log10(Troponina)")
ax.set_ylabel("log10(CK-MB)")
ax.set_title("Estados del MDP según Troponina y CK-MB (K-Means, K=4)")
ax.legend(title="Estado")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grafica_estados_dispersion.png", dpi=200)

# ---- Gráfica 2: % de positivos por estado ----
perfil = df.groupby("estado").agg(pos=("y", "mean"), n=("y", "size"))
fig, ax = plt.subplots(figsize=(7, 4.5))
barras = ax.bar([f"{e}\n{nombres[e]}" for e in perfil.index], perfil["pos"] * 100,
                color=[colores[e] for e in perfil.index])
for b, p, n in zip(barras, perfil["pos"], perfil["n"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"{p*100:.1f}%\n(n={n})", ha="center", fontsize=9)
ax.set_ylabel("% de pacientes positivos")
ax.set_ylim(0, 120)
ax.set_title("Proporción de positivos por estado")
plt.tight_layout()
plt.savefig("grafica_estados_positivos.png", dpi=200)

plt.show()
print("Guardadas: grafica_estados_dispersion.png, grafica_estados_positivos.png")
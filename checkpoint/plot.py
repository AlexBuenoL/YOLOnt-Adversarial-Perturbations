import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO



df = pd.read_csv("outputs/metrics/compression_results.csv")
df["Compression"] = 100 - df["quality"]
plt.figure(figsize=(12, 10))

# 1. Suppression rate
plt.subplot(2, 2, 1)
plt.plot(df["Compression"], df["suppression_rate"], marker="o")
plt.title("Suppression Rate vs Compression")
plt.xlabel("Compression (%)")
plt.ylabel("Suppression Rate")
plt.grid(True)

# 2. Confidence drop
plt.subplot(2, 2, 2)
plt.plot(df["Compression"], df["mean_conf_drop"], marker="o", color="orange")
plt.title("Mean Confidence Drop vs Compression")
plt.xlabel("Compression (%)")
plt.ylabel("Confidence Drop")
plt.grid(True)

# 3. PSNR
plt.subplot(2, 2, 3)
plt.plot(df["Compression"], df["mean_psnr_db"], marker="o", color="green")
plt.title("PSNR vs Compression")
plt.xlabel("Compression (%)")
plt.ylabel("PSNR (dB)")
plt.grid(True)

# 4. SSIM
plt.subplot(2, 2, 4)
plt.plot(df["Compression"], df["mean_ssim"], marker="o", color="red")
plt.title("SSIM vs Compression")
plt.xlabel("Compression (%)")
plt.ylabel("SSIM")
plt.grid(True)

plt.tight_layout()
plt.show()
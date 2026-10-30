# Ganta · script x-FUSegNet

[Python source asli](x-FUSegNet%20Reproduce.py) disalin byte-identik dari working directory partner. Script berisi training lima fold, inference average/voting, evaluasi, dan pembuatan figure. Mengimpor/mengeksekusi script dapat menjalankan rangkaian tersebut; audit publikasi hanya memakai AST dan cache PNG, tanpa mengimpor entrypoint.

Source mengharapkan CWD folder `notebook/` (`BASE_DIR=".."`), raw data di `../data/raw/fuseg_2021/`, dan menulis ke `../output_fuseg_2021/`. **`TEST_DIR` menunjuk `validation`, bukan challenge test.** Lima fold memakai 810 provided train; evaluasi ensemble memakai 200 provided validation.

Environment yang tercatat pada run: PyTorch 2.7.1+cu118, SMP 0.6.0, Albumentations 2.0.8. Full dependency lock dan state RNG run tidak tersedia. Jangan menganggap environment frozen Orie identik dengan environment partner.

Lihat [perbandingan dan batas verifikasi](../reports/ganta-xfusegnet-comparison.md), [hasil asli](../output_fuseg_2021/results/), dan [MIT notice upstream](UPSTREAM_LICENSE.txt). Dua notebook `.ipynb` partner yang tersimpan mengandung training terinterupsi; publikasi ini berfokus pada script `.py` dan hasil yang diminta, tanpa mengeksekusi ulang training.

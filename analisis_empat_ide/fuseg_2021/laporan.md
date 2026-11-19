# Analisis empat ide: fuseg_2021

Evaluasi: `data/raw/fuseg_2021/validation` (200 citra). Luka kecil = luas > 0 dan < 1% citra. Komponen < 20px diabaikan.

Folder prediksi yang dipakai (otomatis: timestamp terbaru yang berisi PNG):

```
baseline: baseline_efficientnet-b7_2026-11-15_13-12-17
cl_dpsc: cl_dpsc_efficientnet-b7_2026-11-06_20-25-49
bw_ftl: bw_ftl_efficientnet-b7_2026-11-15_13-12-14
cascade: cascade_efficientnet-b7_2026-11-14_00-59-20
```

> Satu run, satu seed. Selisih kecil belum bisa diklaim (lihat noise floor di A4).

## A1. Diagnosis kegagalan (model: `baseline`)

Dice dan jenis kegagalan per kelas luas luka (share_of_total_deficit = porsi total kehilangan Dice yang berasal dari kelas itu):

```
            n  dice_mean  dice_median  miss_total  under_seg  fp_heavy  share_of_total_deficit
area_bin                                                                                      
<0.1%      17      0.623        0.746       0.118      0.059     0.235                   0.246
0.1-1%    106      0.862        0.914       0.000      0.057     0.028                   0.563
1-5%       65      0.931        0.957       0.000      0.031     0.000                   0.173
>5%         7      0.934        0.938       0.000      0.000     0.000                   0.018
```

Korelasi Spearman:

```
                               pasangan     rho       p    n
0            Dice vs log10(luas), semua  0.4885  0.0000  195
1             Dice vs kontras dE, semua  0.3893  0.0000  195
2  Dice vs kontras dE, hanya luka kecil  0.2861  0.0013  123
```

Tabel 2x2 ukuran x kontras (batas kontras = median dE 23.6). Kalau 'kontras rendah' jauh lebih buruk di kedua baris, penyebabnya kontras, bukan ukuran:

```
                     n  dice_mean  miss_total
ukuran     kontras                           
besar      rendah   29      0.904       0.000
           tinggi   43      0.949       0.000
kecil(<1%) rendah   68      0.781       0.029
           tinggi   55      0.887       0.000
```

Gambar: `a1_diagnosis.png`

## A2. Batas atas cascade berbasis proposal prediksi (coarse = `baseline`, refiner = `cascade`)

> Catatan: PNG coarse dari cascade tidak disimpan, jadi **baseline dipakai sebagai proksi** model coarse. Beri `--coarse-dir` jika punya PNG coarse yang sebenarnya.

Komponen luka GT yang tidak tersentuh sama sekali oleh coarse (hit = ada piksel prediksi yang overlap):

```
          komponen  terlewat  hit_rate  mean_cover  px_terlewat
size_bin                                                       
<0.1%           41        14     0.659       0.500       1288.0
0.1-1%         151         5     0.967       0.861       2786.0
1-5%            59         1     0.983       0.943       2785.0
>5%              7         0     1.000       0.928          0.0
```

Total komponen GT: 258; terlewat: 20 (7.8%); piksel luka dalam komponen terlewat: 1.07% dari semua piksel luka.

Dari komponen yang terlewat coarse, yang berhasil disentuh cascade: 3/20. Sebaliknya, yang tadinya kena lalu hilang di cascade: 6.

Citra dengan GT tapi **prediksi coarse kosong** (cascade tidak punya crop): 2/195.

Komponen terlewat pada citra yang *punya* prediksi: 18; di luar bbox prediksi (+20% margin): 16. Angka 'di luar' = kasus yang tidak akan terjangkau crop berbasis bbox prediksi (aturan crop asli notebook 13 bisa berbeda, ini perkiraan).

Kehilangan Dice coarse per jenis kegagalan. Kalau `miss_total` dominan, proposal berbasis ketidakpastian punya ruang; kalau `under_seg`/`fp_heavy` dominan, zoom saja sudah cukup:

```
              n  share_of_total_deficit
cat                                    
fp_heavy      7                   0.144
miss_total    2                   0.077
ok          177                   0.586
under_seg     9                   0.193
```

Komponen prediksi yang tidak overlap GT (false positive terpisah):

```
                    coarse  cascade
jumlah komponen FP      24       14
piksel FP            14524    12238
```

## A3. Data train: kolam luka kecil untuk Copy-Paste / oversampling

Citra train berlabel: 791 | citra luka kecil (<1%): 473 (59.8%).

```
          train_n  eval_n  train_%  eval_%  baseline_dice
area_bin                                                 
<0.1%          53      17    6.700   8.718          0.623
0.1-1%        420     106   53.097  54.359          0.862
1-5%          282      65   35.651  33.333          0.931
>5%            36       7    4.551   3.590          0.934
```

Komponen luka (>= 20px): 1031; yang kecil (<1%): 718 dari 586 citra sumber. Ini ukuran 'kolam' yang bisa ditempel (Copy-Paste) ke area kulit lain.

Agar citra luka kecil mencapai 50% dari paparan per epoch, bobot sampling citra luka kecil ~1.00x terhadap citra besar (1.00 berarti sudah di atas target).

> Catatan: ini seluruh folder `train` (sebelum split 85/15). Dan baca kolom `baseline_dice` bersama `train_n`: kalau kelas kecil punya contoh train yang cukup tapi Dice tetap rendah, masalahnya bukan kelangkaan data, dan oversampling kecil kemungkinan membantu.

Gambar: `a3_sebaran_ukuran.png`

## A4. Perbandingan berpasangan vs baseline (per citra, hanya citra ber-GT)

`mean_delta` = rata-rata (Dice model - Dice baseline) per citra; CI = bootstrap 95%; p = Wilcoxon berpasangan. Dengan banyak subset dan model, p kecil di satu-dua baris bisa kebetulan, jadi cari pola yang konsisten, bukan satu angka.

### cl_dpsc vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           195       0.010    0.002    0.020       0.073   110      84
kecil(<1%)      123       0.016    0.003    0.032       0.083    66      56
besar            72      -0.001   -0.007    0.006       0.321    44      28
bin <0.1%        17       0.026   -0.036    0.100       0.959     9       7
bin 0.1-1%      106       0.015    0.004    0.029       0.045    57      49
bin 1-5%         65      -0.001   -0.008    0.006       0.322    41      24
bin >5%           7      -0.000   -0.020    0.016       0.938     3       4
kontras_sedang   65       0.004   -0.004    0.011       0.080    41      23
kontras_rendah   65       0.023    0.000    0.053       0.238    32      33
kontras_tinggi   65       0.003   -0.005    0.013       0.727    37      28
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan cl_dpsc: 1; sebaliknya baru gagal total di cl_dpsc: 0.

### bw_ftl vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           195       0.008   -0.001    0.019       0.056   110      85
kecil(<1%)      123       0.012   -0.000    0.029       0.153    70      53
besar            72      -0.000   -0.008    0.007       0.100    40      32
bin <0.1%        17       0.047   -0.036    0.158       0.747     8       9
bin 0.1-1%      106       0.007    0.001    0.012       0.055    62      44
bin 1-5%         65      -0.001   -0.009    0.007       0.206    35      30
bin >5%           7       0.005   -0.010    0.018       0.375     5       2
kontras_sedang   65       0.011   -0.006    0.039       0.863    33      32
kontras_rendah   65       0.013   -0.003    0.030       0.144    41      24
kontras_tinggi   65      -0.001   -0.008    0.005       0.116    36      29
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan bw_ftl: 2; sebaliknya baru gagal total di bw_ftl: 0.

### cascade vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           195       0.004   -0.008    0.016       0.292   100      93
kecil(<1%)      123       0.006   -0.012    0.026       0.199    67      54
besar            72      -0.001   -0.006    0.004       0.809    33      39
bin <0.1%        17      -0.038   -0.134    0.035       0.307     5      10
bin 0.1-1%      106       0.013   -0.002    0.031       0.035    62      44
bin 1-5%         65      -0.002   -0.007    0.003       0.524    28      37
bin >5%           7       0.007   -0.004    0.018       0.297     5       2
kontras_sedang   65      -0.000   -0.011    0.010       0.316    35      29
kontras_rendah   65       0.010   -0.021    0.044       0.635    31      33
kontras_tinggi   65       0.001   -0.006    0.008       0.466    34      31
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan cascade: 0; sebaliknya baru gagal total di cascade: 2.

**Noise floor** (arsitektur identik, seed/run berbeda: `cascade_coarse` vs `baseline`, dari CSV; kolom tebakan nama=`Name`, dice=`Dice`/`Dice`):

```
     n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
0  200       0.549   -0.321    1.851       0.134   109      85
```

Selisih model lain yang tidak jauh lebih besar dari angka ini belum bisa diklaim.

Gambar: `a4_delta_per_kelas.png`

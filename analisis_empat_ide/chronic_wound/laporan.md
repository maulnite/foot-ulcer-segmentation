# Analisis empat ide: chronic_wound

Evaluasi: `data/raw/chronic_wound/test` (278 citra). Luka kecil = luas > 0 dan < 1% citra. Komponen < 20px diabaikan.

Folder prediksi yang dipakai (otomatis: timestamp terbaru yang berisi PNG):

```
baseline: baseline_chronic_wound_efficientnet-b7_2026-11-10_15-10-52
cl_dpsc: cl_dpsc_chronic_wound_efficientnet-b7_2026-11-08_13-22-01
bw_ftl: bw_ftl_chronic_wound_efficientnet-b7_2026-11-10_15-08-22
cascade: cascade_chronic_wound_efficientnet-b7_2026-11-14_00-59-20
```

> Satu run, satu seed. Selisih kecil belum bisa diklaim (lihat noise floor di A4).

## A1. Diagnosis kegagalan (model: `baseline`)

Dice dan jenis kegagalan per kelas luas luka (share_of_total_deficit = porsi total kehilangan Dice yang berasal dari kelas itu):

```
            n  dice_mean  dice_median  miss_total  under_seg  fp_heavy  share_of_total_deficit
area_bin                                                                                      
<0.1%       3      0.608        0.723         0.0      0.000     0.333                   0.037
0.1-1%     68      0.832        0.880         0.0      0.044     0.044                   0.361
1-5%      145      0.892        0.931         0.0      0.048     0.000                   0.496
>5%        45      0.925        0.967         0.0      0.022     0.000                   0.106
```

Korelasi Spearman:

```
                               pasangan     rho       p    n
0            Dice vs log10(luas), semua  0.5066  0.0000  261
1             Dice vs kontras dE, semua  0.3707  0.0000  261
2  Dice vs kontras dE, hanya luka kecil  0.3206  0.0064   71
```

Tabel 2x2 ukuran x kontras (batas kontras = median dE 22.5). Kalau 'kontras rendah' jauh lebih buruk di kedua baris, penyebabnya kontras, bukan ukuran:

```
                     n  dice_mean  miss_total
ukuran     kontras                           
besar      rendah   91      0.870         0.0
           tinggi   99      0.927         0.0
kecil(<1%) rendah   39      0.782         0.0
           tinggi   32      0.872         0.0
```

Gambar: `a1_diagnosis.png`

## A2. Batas atas cascade berbasis proposal prediksi (coarse = `baseline`, refiner = `cascade`)

> Catatan: PNG coarse dari cascade tidak disimpan, jadi **baseline dipakai sebagai proksi** model coarse. Beri `--coarse-dir` jika punya PNG coarse yang sebenarnya.

Komponen luka GT yang tidak tersentuh sama sekali oleh coarse (hit = ada piksel prediksi yang overlap):

```
          komponen  terlewat  hit_rate  mean_cover  px_terlewat
size_bin                                                       
<0.1%            8         4     0.500       0.475        143.0
0.1-1%          91         8     0.912       0.772       1237.0
1-5%           151         0     1.000       0.882          0.0
>5%             41         0     1.000       0.910          0.0
```

Total komponen GT: 291; terlewat: 12 (4.1%); piksel luka dalam komponen terlewat: 0.35% dari semua piksel luka.

Dari komponen yang terlewat coarse, yang berhasil disentuh cascade: 3/12. Sebaliknya, yang tadinya kena lalu hilang di cascade: 2.

Citra dengan GT tapi **prediksi coarse kosong** (cascade tidak punya crop): 0/261.

Komponen terlewat pada citra yang *punya* prediksi: 12; di luar bbox prediksi (+20% margin): 9. Angka 'di luar' = kasus yang tidak akan terjangkau crop berbasis bbox prediksi (aturan crop asli notebook 13 bisa berbeda, ini perkiraan).

Kehilangan Dice coarse per jenis kegagalan. Kalau `miss_total` dominan, proposal berbasis ketidakpastian punya ruang; kalau `under_seg`/`fp_heavy` dominan, zoom saja sudah cukup:

```
             n  share_of_total_deficit
cat                                   
fp_heavy     4                   0.065
ok         246                   0.704
under_seg   11                   0.232
```

Komponen prediksi yang tidak overlap GT (false positive terpisah):

```
                    coarse  cascade
jumlah komponen FP      20       13
piksel FP             2649     2599
```

## A3. Data train: kolam luka kecil untuk Copy-Paste / oversampling

Citra train berlabel: 779 | citra luka kecil (<1%): 232 (29.8%).

```
          train_n  eval_n  train_%  eval_%  baseline_dice
area_bin                                                 
<0.1%          13       3    1.669   1.149          0.608
0.1-1%        219      68   28.113  26.054          0.832
1-5%          411     145   52.760  55.556          0.892
>5%           136      45   17.458  17.241          0.925
```

Komponen luka (>= 20px): 841; yang kecil (<1%): 287 dari 276 citra sumber. Ini ukuran 'kolam' yang bisa ditempel (Copy-Paste) ke area kulit lain.

Agar citra luka kecil mencapai 50% dari paparan per epoch, bobot sampling citra luka kecil ~2.36x terhadap citra besar (1.00 berarti sudah di atas target).

> Catatan: ini seluruh folder `train` (sebelum split 85/15). Dan baca kolom `baseline_dice` bersama `train_n`: kalau kelas kecil punya contoh train yang cukup tapi Dice tetap rendah, masalahnya bukan kelangkaan data, dan oversampling kecil kemungkinan membantu.

Gambar: `a3_sebaran_ukuran.png`

## A4. Perbandingan berpasangan vs baseline (per citra, hanya citra ber-GT)

`mean_delta` = rata-rata (Dice model - Dice baseline) per citra; CI = bootstrap 95%; p = Wilcoxon berpasangan. Dengan banyak subset dan model, p kecil di satu-dua baris bisa kebetulan, jadi cari pola yang konsisten, bukan satu angka.

### cl_dpsc vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           261       0.000   -0.012    0.011       0.075   151     109
kecil(<1%)       71      -0.020   -0.055    0.006       0.786    42      28
besar           190       0.008   -0.002    0.018       0.070   109      81
bin <0.1%         3      -0.067   -0.179    0.011       0.500     1       2
bin 0.1-1%       68      -0.018   -0.053    0.008       0.570    41      26
bin 1-5%        145       0.010   -0.000    0.023       0.084    84      61
bin >5%          45      -0.001   -0.019    0.012       0.531    25      20
kontras_sedang   87       0.008   -0.003    0.019       0.035    53      34
kontras_rendah   87      -0.010   -0.043    0.019       0.722    47      40
kontras_tinggi   87       0.002   -0.004    0.010       0.091    51      35
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan cl_dpsc: 0; sebaliknya baru gagal total di cl_dpsc: 2.

### bw_ftl vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           261      -0.002   -0.012    0.007       0.911   125     136
kecil(<1%)       71      -0.004   -0.033    0.020       0.602    30      41
besar           190      -0.001   -0.009    0.006       0.633    95      95
bin <0.1%         3       0.141    0.051    0.293       0.250     3       0
bin 0.1-1%       68      -0.010   -0.041    0.014       0.271    27      41
bin 1-5%        145      -0.002   -0.012    0.007       0.864    69      76
bin >5%          45       0.001   -0.011    0.013       0.238    26      19
kontras_sedang   87      -0.006   -0.020    0.005       0.591    38      49
kontras_rendah   87      -0.004   -0.032    0.017       0.993    41      46
kontras_tinggi   87       0.005   -0.003    0.014       0.542    46      41
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan bw_ftl: 0; sebaliknya baru gagal total di bw_ftl: 3.

### cascade vs baseline

```
                  n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
subset                                                                     
semua           261       0.022    0.011    0.034       0.000   159     102
kecil(<1%)       71       0.014   -0.008    0.034       0.005    46      25
besar           190       0.024    0.012    0.040       0.003   113      77
bin <0.1%         3      -0.001   -0.134    0.127       1.000     2       1
bin 0.1-1%       68       0.015   -0.008    0.034       0.004    44      24
bin 1-5%        145       0.025    0.009    0.043       0.026    83      62
bin >5%          45       0.024    0.006    0.043       0.026    30      15
kontras_sedang   87       0.018    0.004    0.035       0.008    57      30
kontras_rendah   87       0.036    0.010    0.066       0.000    58      29
kontras_tinggi   87       0.011   -0.000    0.024       0.556    44      43
```

Citra yang di baseline gagal total (Dice-nya miss_total) tapi dipulihkan cascade: 0; sebaliknya baru gagal total di cascade: 2.

**Noise floor** (arsitektur identik, seed/run berbeda: `cascade_coarse` vs `baseline`, dari CSV; kolom tebakan nama=`Name`, dice=`Dice`/`Dice`):

```
     n  mean_delta  ci95_lo  ci95_hi  p_wilcoxon  wins  losses
0  278       0.023   -1.793    1.809       0.398   137     128
```

Selisih model lain yang tidak jauh lebih besar dari angka ini belum bisa diklaim.

Gambar: `a4_delta_per_kelas.png`

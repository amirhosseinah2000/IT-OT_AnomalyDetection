# راهنمای نمودارهای خروجی مدل و مزیت AI

این راهنما نقشهٔ نمودارهای اضافه‌شده به «مدل‌ها» و «تابلوی کل سامانه» است. همهٔ نمودارها artifact-first هستند: دادهٔ واقعی از همان run خوانده می‌شود و فایل‌های پروتکل‌ها با هم یکی نمی‌شوند. آموزش جدید همهٔ artifactهای ارزیابی Stage 2 و end-to-end این راهنما را می‌سازد؛ runهای قدیمی که پیش از این نسخه ساخته شده‌اند، فقط نام دقیق فایلِ لازم را نشان می‌دهند و باید یک‌بار دوباره آموزش داده شوند.

## مسیر نمایش

- «مدل‌ها → اولویت: مزیت AI»: توضیح خروجی چندبعدی و ۲۵ نمای قابل انتخاب برای هر نوع انومالی.
- «مدل‌ها → ارزیابی Stage 1 (۱۵ نمودار)»: تشخیص باینری LSTM-AE + Isolation Forest.
- «مدل‌ها → ارزیابی Stage 2 (۱۵ نمودار)»: طبقه‌بندی نوع حمله با Random Forest.
- «مدل‌ها → ارزیابی end-to-end (۱۵ نمودار)»: اثر واقعی گیت Stage 1 بر نتیجهٔ نهایی.
- «تابلوی کل سامانه → مقایسهٔ مدل‌ها و AI»: مقایسهٔ جدیدترین مدل هر پروتکل. اگر فقط HTTP آموزش دیده باشد، دقیقاً همان محدودیت در جدول و نمودارها دیده می‌شود.

## ۲۵ نمای هر نوع انومالی

| گروه | شماره‌ها و نمودارها | منبع و تفسیر درست |
|---|---|---|
| فضای یادگرفته‌شده | ۱ جداسازی t-SNE، ۲ تعامل دو فیچر، ۳ مسیر embedding، ۴ امضای چندبعدی، ۵ فیچرهای شکل‌دهنده | `stage1/evaluation-prepared.parquet`، `feature-evidence.parquet` و مدل-space. این‌ها نشان‌دهندهٔ الگوی چندمتغیره‌اند، نه اثبات رابطهٔ علّی. |
| زمان و پایداری | ۶ score AI در برابر z تک‌فیچر، ۷ باند تطبیقی، ۸ زمان هشدار AI/قانون ثابت، ۹ jitter، ۱۰ alert strip | `stage1/scores.parquet` و `timestamp`. زمان هشدار از شروع capture اندازه‌گیری می‌شود؛ زمان واقعی آغاز حمله در CSV موجود نیست. |
| تفسیرپذیری | ۱۱ heatmap شواهد زمان×فیچر، ۱۲ همبستگی متحرک، ۱۳ waterfall رخداد، ۱۴ نزدیک‌ترین نرمال، ۱۵ فراوانی شواهد | `stage1/feature-evidence.parquet` و نرمال آماده‌شده. Waterfall فاصلهٔ robust از baseline است و SHAP یا علت قطعی شبکه نیست. |
| عملیات و نوظهوری | ۱۶ gauge، ۱۷ فاصلهٔ OOD از نزدیک‌ترین نرمال، ۱۸ small multiples، ۱۹ بار/شدت، ۲۰ انتشار بین گره‌ها | scoreهای Stage 1 و فیلدهای PCAP مانند `src_ip`/`dst_ip`. OOD فاصله در فضای مدل است، نه نام‌گذاری نوع حملهٔ ناشناخته. |
| هشدار و بازیابی | ۲۱ شواهد پرتکرار، ۲۲ پایداری IF زیر نویز ۳٪، ۲۳ recovery بعد از اوج، ۲۴ توالی نرمال→شدید، ۲۵ پروفایل شدت اقدام | scoreها، evidence و Isolation Forest فریز‌شده. تست نویز فقط پایداری مؤلفهٔ IF را نشان می‌دهد، نه تضمین مقاوم‌بودن کل سامانه. |

## ارزیابی Stage 1: ۱۵ نمودار و جدول‌ها

| شماره | نمودار | artifact اصلی |
|---|---|---|
| ۱–۵ | confusion matrix، ROC، Precision–Recall، threshold sweep، FPR-vs-Recall | `stage1/scores.parquet` و `stage1/metrics.json` |
| ۶–۸ | توزیع score، خطای بازسازی LSTM، امتیاز Isolation Forest | `stage1/scores.parquet` |
| ۹ | train/validation loss | `stage1/lstm-training-history.parquet` |
| ۱۰ | learning curve بر حسب حجم آموزش | `stage1/learning-curve.parquet`؛ refitهای کنترل‌شده با سقف epoch مشخص‌اند و مدل deployed نیستند |
| ۱۱–۱۲ | نرخ واقعی حمله بر حسب bucket score، توافق LSTM/IF | `stage1/scores.parquet` |
| ۱۳ | Quantile/MAD/Final threshold | `stage1/metrics.json` |
| ۱۴ | پایداری FPR در طول زمان | `stage1/scores.parquet` با timestamp |
| ۱۵ | دسته‌بندی FP/FN | score، تصمیم و رأی detectorها؛ یک ریشه‌یابی تحلیلی است، نه علت قطعی شبکه |

جدول‌های همراه شامل TP/FP/TN/FN و Precision/Recall/F1/FPR/Specificity/AUC/AP، اجزای threshold و شمارش توافق ensemble هستند. در Stage 1، score نرمال‌شده «احتمال کالیبره‌شده» نیست؛ نمودار ۱۱ فقط رابطهٔ تجربی score با فراوانی واقعی حمله را گزارش می‌کند.

## ارزیابی Stage 2: ۱۵ نمودار و جدول‌ها

| شماره | نمودار | artifact اصلی |
|---|---|---|
| ۱۶–۱۸ | confusion matrix چندکلاسه، Precision/Recall/F1 هر کلاس، Macro-vs-Weighted F1 | `stage2/confusion-matrix.parquet` |
| ۱۹ | ROC یک-در-برابر-بقیه | `stage2/probabilities.parquet`؛ احتمال هر کلاس RF و فقط نتیجهٔ held-out |
| ۲۰ | اهمیت Gini Random Forest | `stage2/feature-importance.parquet` |
| ۲۱ | Permutation-vs-Gini | `stage2/permutation-importance.parquet`، با metric `weighted_f1` روی held-out |
| ۲۲–۲۳ | Support-vs-accuracy و زوج‌های اشتباه | confusion matrix و `stage2/scores.parquet` |
| ۲۴–۲۵ | n_estimators sweep و learning curve | `stage2/n-estimators-sweep.parquet` و `stage2/learning-curve.parquet`؛ sweepهای diagnostic با سقف درخت مستقل از RF deployed اجرا می‌شوند |
| ۲۶–۲۷ | confidence درست/غلط و calibration | `stage2/probabilities.parquet` |
| ۲۸ | سهم `stage1_lstm_score` و `stage1_isolation_forest_score` | `stage2/feature-importance.parquet`؛ فقط وقتی این دو feature در قرارداد فعال باشند |
| ۲۹–۳۰ | پایداری CV و sweep `min_samples_leaf` | `stage2/cross-validation.parquet` و `stage2/min-samples-leaf-sweep.parquet`؛ CV فقط داخل بخش train است تا test نهایی حفظ شود |

جدول‌ها: classification report هر کلاس، confusion عددی، ۱۰ فیچر مهم، پیکربندی کلاس/feature و فهرست زوج‌های پرتکرارِ خطا. Weighted F1 می‌تواند عملکرد کلاس‌های پرتعداد را بزرگ‌تر از کلاس‌های نادر جلوه دهد؛ Macro F1 را همیشه کنارش بخوانید.

## ارزیابی end-to-end: ۱۵ نمودار و جدول‌ها

| شماره | نمودار | تفسیر و منبع |
|---|---|---|
| ۳۱–۳۳ | پنج حالت خروجی، funnel، Precision/Recall/F1 نهایی | `pipeline/end-to-end-predictions.parquet`. حمله فقط وقتی موفق است که Stage 1 آن را عبور دهد و Stage 2 نوع درست بدهد. |
| ۳۴ | ablation دومرحله‌ای/تک‌مدل | `pipeline/ablation-comparison.parquet`؛ هر دو مدل روی همان ترکیب held-out از نرمال validation و attack-test ارزیابی می‌شوند. |
| ۳۵–۳۶ | missed attack بر حسب نوع و فراخوانی بی‌فایدهٔ Stage 2 | end-to-end predictions. |
| ۳۷ | latency هر مرحله | `pipeline/inference-latency.parquet`؛ زمان کل آموزش جای latency inference نیست. |
| ۳۸ | زمان تا هشدار | proxy از اولین packet capture تا اولین هشدار؛ timestamp واقعی شروع رخداد لازم است. |
| ۳۹–۴۰ | sensitivity threshold و ROC/PR نهایی | `pipeline/threshold-sensitivity.parquet` و `pipeline/end-to-end-probabilities.parquet`. برای هر threshold، Stage 2 روی candidateهای held-out دوباره score می‌شود؛ موفقیت نهایی یعنی «کشف حمله + نوع صحیح». |
| ۴۱–۴۲ | هزینهٔ وزن‌دار خطا و سطح اطمینان alert | predictions؛ وزن‌های هزینه فرض عملیاتی شفاف‌اند و قابل تغییرند. |
| ۴۳–۴۴ | پایداری seed و records/sec | `pipeline/seed-stability.parquet` و `pipeline/inference-latency.parquet`. در پایداری seed، Stage 1 عمداً ثابت است و فقط RF Stage 2 با seedهای متفاوت بازآموزی می‌شود؛ این محدودیت در عنوان نمودار مشخص است. |
| ۴۵ | روند نرخ خطای کل | end-to-end predictions دارای timestamp. |

## مقایسهٔ همهٔ پروتکل‌ها

این نما حداقل نمودارهای ۲۸ تا ۵۲ سند ارزیابی را پوشش می‌دهد: معیارهای گروهی و heatmap F1، ROC/PR overlay، radar، نرخ anomaly و FPR زمانی، هم‌پوشانی/treemap/واگرایی اهمیت feature، ترکیب normal/attack و Sankey protocol→attack-type، مقایسهٔ IT/OT، funnel، RAM، PSI، ماتریس توصیه، ridge/violin score، حباب ریسک، توافق ensemble و small-multiples confusion matrix.

جدول‌های تجمیعی عبارت‌اند از Master Comparison، هم‌پوشانی Top-10 feature، ترکیب dataset، PSI، پیشنهاد مدل، منابع و سلول‌های confusion. معیارهای latency/throughput از `pipeline/inference-latency.parquet` همان مدل خوانده می‌شوند؛ داشبورد هرگز زمان آموزش را به‌جای latency جا نمی‌زند.

## استفاده بیرون از داشبورد

همان جدول‌ها از Parquet/JSONهای زیر قابل‌خواندن‌اند و CSV دانلودی صرفاً نسخهٔ قابل انتقال همان جدول نمایش‌داده‌شده است:

```text
<run>/models/<protocol>-<profile>/stage1/scores.parquet
<run>/models/<protocol>-<profile>/stage1/feature-evidence.parquet
<run>/models/<protocol>-<profile>/stage1/lstm-training-history.parquet
<run>/models/<protocol>-<profile>/stage2/scores.parquet
<run>/models/<protocol>-<profile>/stage2/confusion-matrix.parquet
<run>/models/<protocol>-<profile>/stage2/feature-importance.parquet
<run>/models/<protocol>-<profile>/pipeline/end-to-end-predictions.parquet
<run>/models/<protocol>-<profile>/training-summary.json
```

در هر integration، `protocol`، `capture`، `record_id`، `mapping_accepted` و timezone UTC را حفظ کنید. برای سنجش مدل فقط recordهای با نگاشت پذیرفته‌شده را در Stage 2 وارد کنید.

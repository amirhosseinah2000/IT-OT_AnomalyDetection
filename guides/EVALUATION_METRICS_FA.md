# راهنمای معیارهای ارزیابی سامانهٔ دو مرحله‌ای

این راهنما معیارهایی را تعریف می‌کند که dashboard، `training-summary.json`، `stage1/metrics.json` و `stage2/metrics.json` نمایش می‌دهند. هدف، گزارش درست عملکرد است؛ مخصوصاً چون در این پروژه false positive مرحلهٔ اول هزینهٔ عملیاتی بالایی دارد.

## پیش‌نیاز اعتبار ارزیابی

قبل از خواندن هر عدد مدل، این سه شرط را بررسی کنید:

1. featureها فقط از PCAP استخراج شده‌اند و profile/pipeline همان run استفاده شده است.
2. رکوردهای حمله فقط وقتی وارد آموزش Stage 2 می‌شوند که `mapping_accepted=true` باشد؛ acceptance به match rate و mean match confidence وابسته است.
3. LSTM-AE و Isolation Forest فقط با بخش آموزش ترافیک benign fit می‌شوند. partition اعتبارسنجی benign برای threshold نگه داشته می‌شود.

اگر یکی از این شرط‌ها برقرار نباشد، یک F1 یا accuracy ظاهراً بالا قابل استقرار نیست.

## ماتریس خطای Stage 1

Stage 1 پاسخ می‌دهد: «آیا این packet/record باید anomaly تلقی و به Stage 2 فرستاده شود؟» برچسب مثبت = حمله/anomaly و برچسب منفی = نرمال.

| نماد | معنی عملی |
|---|---|
| TP | حمله‌ای که anomaly تشخیص داده شد. |
| FP | ترافیک نرمالی که اشتباه anomaly شد؛ موجب هشدار یا ارسال بی‌مورد به Stage 2 می‌شود. |
| TN | ترافیک نرمالی که درست نرمال ماند. |
| FN | حمله‌ای که از گیت عبور نکرد؛ خطرناک‌ترین خطا برای پوشش حمله. |

| معیار | فرمول | چه سؤال را پاسخ می‌دهد | تفسیر در این سامانه |
|---|---|---|---|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | چند تصمیم کلّی درست بود؟ | در دادهٔ نامتوازن می‌تواند گمراه‌کننده باشد؛ به‌تنهایی معیار پذیرش نیست. |
| Precision | `TP/(TP+FP)` | از هشدارها چند مورد واقعاً حمله‌اند؟ | Precision پایین یعنی تحلیلگر با هشدارهای اشتباه درگیر می‌شود. |
| Recall / TPR | `TP/(TP+FN)` | چند حمله پیدا شده‌اند؟ | Recall پایین یعنی Stage 2 هرگز بسیاری از حمله‌ها را نمی‌بیند. |
| F1 | `2×Precision×Recall/(Precision+Recall)` | تعادل Precision و Recall چیست؟ | جمع‌بندی مفید است، اما FPR باید جداگانه دیده شود. |
| Specificity / TNR | `TN/(TN+FP)` | چند نرمال درست عبور کردند؟ | مکمل FPR است؛ برای سامانهٔ عملیاتی مهم است. |
| FPR | `FP/(FP+TN)` | چه سهمی از نرمال‌ها هشدار اشتباه شدند؟ | معیار اولویت‌دار این پروژه. FPR کم با Recall خیلی پایین موفقیت نیست. |
| FNR | `FN/(FN+TP)` | چه سهمی از حمله‌ها از دست رفتند؟ | مکمل Recall؛ کنار FPR گزارش شود. |
| ROC-AUC | مساحت زیر منحنی TPR در برابر FPR در آستانه‌های مختلف | آیا scoreها مثبت/منفی را رتبه‌بندی می‌کنند؟ | نیازمند هر دو کلاس واقعی است؛ بدون label قابل محاسبه نیست. |
| Average precision / PR-AUC | خلاصهٔ Precision–Recall روی thresholdها | کیفیت رتبه‌بندی در کلاس مثبت کم‌یاب چیست؟ | برای حملهٔ کم‌تعداد معمولاً از ROC خواناتر است. |

## threshold و ensemble Stage 1

دو detector score تولید می‌کنند:

- `lstm_reconstruction_score`: خطای بازسازی LSTM-AE؛ هرچه بالاتر، رکورد از الگوی benign دورتر است.
- `isolation_forest_score`: منفی `score_samples` Isolation Forest؛ هرچه بالاتر، منزوی‌تر/ناهنجارتر است.

برای هر score، threshold از validation benign ساخته می‌شود: بزرگ‌ترینِ quantile هدف FPR و guardrail `median + 3.5×1.4826×MAD`. سپس mode پیکربندی‌شده تعیین می‌کند `LSTM OR Forest` یا `LSTM AND Forest` anomaly باشد. `stage1_normalized_score` بیشینهٔ نسبت score به thresholdهای دو detector است؛ مقدار حدود 1 مرز گیت است.

| رفتار threshold | اثر محتمل |
|---|---|
| پایین‌تر | Recall بالاتر، ولی FPR و حجم هشدار بالاتر. |
| بالاتر | FPR پایین‌تر، ولی FN/حملهٔ از دست‌رفته بالاتر. |
| OR ensemble | حساس‌تر؛ احتمال Recall بیشتر و FPR بیشتر. |
| AND ensemble | محافظه‌کارتر؛ احتمال FPR کمتر و FN بیشتر. |

هر تغییر profile، دادهٔ benign، threshold یا ensemble mode نیازمند ارزیابی مجدد است؛ threshold قبلی را روی مدل/profile جدید حمل نکنید.

## معیارهای Stage 2: Random Forest نوع حمله

Stage 2 فقط ردیف‌های حمله با mapping پذیرفته‌شده را برای آموزش می‌بیند و در inference فقط anomalyهای عبورکرده از Stage 1 را نوع‌بندی می‌کند.

| معیار | فرمول/روش | معنی |
|---|---|---|
| Test accuracy | تعداد پیش‌بینی صحیح تقسیم بر recordهای test | دقت کلی classهای حمله در test split. |
| Weighted F1 | میانگین F1 هر class با وزن support آن class | معیار اصلی ثبت‌شده برای Stage 2؛ اثر classهای پرتکرار را متناسب نگه می‌دارد. |
| Confusion matrix | ردیف = نوع حملهٔ واقعی، ستون = نوع پیش‌بینی‌شده | مشخص می‌کند کدام دو نوع حمله با هم اشتباه می‌شوند. |
| Class support | تعداد record هر label | برای تشخیص F1های ناپایدار در classهای کوچک ضروری است. |
| Feature importance | `feature_importances_` Random Forest | سهم نسبی splitهای درخت‌ها؛ علت قطعی یا رابطهٔ علّی نیست. |

`class_weight='balanced_subsample'` برای کم‌کردن اثر عدم‌توازن درخت‌ها استفاده می‌شود، اما جایگزین دادهٔ کافی برای class کوچک نیست. در class با فقط چند record، weighted F1 می‌تواند خوب به‌نظر برسد و پوشش class نادر را پنهان کند؛ confusion matrix و support را همواره کنار آن بخوانید.

## معیارهای داده و نگاشت

| معیار | محل | تعریف/کاربرد |
|---|---|---|
| Match rate | mapping audit | نسبت Flow/recordهای PCAP که با evidence زمان و endpoint به CSV وصل شده‌اند. |
| Mean match confidence | mapping audit | میانگین confidence تصمیم mapping؛ quality هم‌زمان با match rate خوانده می‌شود. |
| Mapping accepted | mapping audit | true فقط اگر match rate و confidence از حداقل‌های config عبور کنند. |
| Selected offset seconds | mapping audit | offset زمانی انتخاب‌شده برای هم‌ترازکردن ساعت CSV/PCAP، با evidence endpoint. |
| Pairing score | mapping audit | شباهت اولیهٔ نام فایل؛ معیار پذیرش واقعی نیست. |
| Class distribution | attack label distribution | شمار PCAP-derived recordهای class برای برنامهٔ balancing و split. |

پیش‌فرض حداقل‌ها در config عبارت‌اند از `minimum_match_rate=0.90` و `minimum_match_confidence=0.90`. این‌ها قاعدهٔ ورود Stage 2 هستند، نه معیار performance مدل.

## معیارهای کیفیت feature و پیش‌پردازش

| معیار | تعریف | تصمیمی که پشتیبانی می‌کند |
|---|---|---|
| Availability / observed ratio | نسبت ردیف‌های دارای مقدار غیرخالی | feature با پوشش خیلی پایین احتمالاً قبل از مدل حذف می‌شود. |
| Missing ratio | `1 - observed ratio` | تفاوت NaN واقعی و صفر را روشن می‌کند. |
| Zero ratio | سهم مقدارهای صفر | feature sparse/تقریباً ثابت را مشخص می‌کند. |
| Unique count | تعداد مقدار یکتا | cardinality بالا می‌تواند one-hot را بزرگ کند. |
| Variance / std / IQR | پراکندگی عددی feature | feature ثابت یا کم‌اطلاع را آشکار می‌کند. |
| Model usable | نتیجهٔ quality report | آیا feature واقعاً استخراج شده و برای pipeline مناسب است؟ |
| Raw vs prepared IQR/std | preprocessing comparison | اثر imputation، robust scaling و clipping را قبل از آموزش قابل مشاهده می‌کند. |
| Transformed input count | manifest/summary | تعداد ستون واقعی مدل پس از one-hot؛ با تعداد feature profile فرق دارد. |

قانون حذف پیش‌فرض: non-null ratio کمتر از 5٪ یا `nunique(dropna=True) <= 1` باعث حذف feature از contract می‌شود. مقدارهای numeric با median benign-imputation، robust scaling و clip `[-12,+12]` آماده می‌شوند؛ categoricalها با most-frequent و one-hot تبدیل می‌شوند.

## معیارهای منابع و زمان

| معیار | منبع | معنی و محدودیت |
|---|---|---|
| Training seconds | `training-summary.json` | زمان کل همان run مدل از آغاز تا ذخیرهٔ artifactها. بین دستگاه‌های متفاوت مستقیم مقایسه نشود. |
| Process RSS MB | `reports/resource-usage.parquet` | حافظهٔ resident process پیش/پس از آموزش؛ peak لحظه‌ای نیست. |
| System memory percent / available GB | resource usage و مانیتور زنده | سلامت ظرفیت میزبان در لحظهٔ snapshot. |
| CPU percent / logical CPUs | مانیتور منابع dashboard | مصرف لحظه‌ای میزبان، نه زمان CPU صرفاً متعلق به مدل. |
| Input features / rows / windows | summary، manifest و LSTM history | توضیح تفاوت زمان/حافظه میان profileها و runها. |
| Train/validation loss | `lstm-training-history.parquet` | روند بازسازی benign. اختلاف زیاد validation/train می‌تواند نشانهٔ overfit یا تغییر توزیع باشد. |

## ترتیب پیشنهادی تصمیم برای استقرار

1. mapping audit را بررسی کنید؛ فقط captureهای accepted در آموزش Stage 2 باشند.
2. quality report و raw/prepared comparison را ببینید؛ featureهای NaN، ثابت یا بسیار sparse را profile نکنید.
3. در Stage 1 ابتدا FPR، سپس Recall/FNR، سپس Precision/F1 و ROC/PR را بخوانید.
4. در Stage 2 weighted F1 را با confusion matrix و class support تأیید کنید.
5. زمان آموزش و RSS را با سقف CPU/RAM محیط مقصد مقایسه کنید.
6. model contract و pipeline/manifest همان run را همراه ONNX یا artifact اصلی deploy کنید.

مسیر اصلی معیارها:

```text
<model>/training-summary.json
<model>/stage1/metrics.json
<model>/stage1/scores.parquet
<model>/stage1/lstm-training-history.parquet
<model>/stage2/metrics.json
<model>/stage2/confusion-matrix.parquet
<model>/stage2/feature-importance.parquet
<model>/reports/resource-usage.parquet
<run>/reports/mapping-audit.parquet
<run>/reports/features/*-quality.parquet
```

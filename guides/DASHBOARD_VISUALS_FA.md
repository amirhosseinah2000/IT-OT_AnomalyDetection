# راهنمای نمودارها و جدول‌های داشبورد

این راهنما می‌گوید هر visual چه سؤال مشخصی را پاسخ می‌دهد، داده‌اش از کجا می‌آید و چه چیزی را **نمی‌توان** از آن نتیجه گرفت. نام فایل‌ها و جدول‌های منبع از catalog اجرای انتخاب‌شده قابل ردیابی‌اند؛ dashboard دادهٔ جدید یا عدد ساختگی تولید نمی‌کند.

## قرارداد مشترک

- نمودارهای packet-level برای پاسخ‌گویی سریع از نمونهٔ یکنواخت و محدود `records.parquet` رسم می‌شوند. عنوان آن‌ها «نمونه» را مشخص می‌کند. جدول‌های catalog، نگاشت، کیفیت و model summary مستقیماً از artifactهای ذخیره‌شده خوانده می‌شوند.
- زمان در «تابلوی کل سامانه» UTC است. اندازهٔ bucket با طول بازهٔ مشاهده‌شده تنظیم می‌شود: ساعت، 6 ساعت یا روز.
- آبی معمولاً نرمال، زرشکی حمله/نیاز به بررسی، سبز پذیرفته/درست، بنفش مسیر مدل یا feature و قرمز هشدار/خطا است.
- همهٔ نمودارها در سطح اجرای دیتاست انتخاب‌شده‌اند. برای جزئیات یک پروتکل، آن را از سایدبار انتخاب کنید؛ فایل‌های خام همچنان جدا هستند و aggregate باعث یکی‌شدن فایل‌های feature نمی‌شود.

## تابلوی کل سامانه: نمودارها

این بخش در منوی «تابلوی کل سامانه» همیشه همهٔ پروتکل‌های run را می‌بیند. اگر artifact لازم موجود نباشد، visual وابسته به آن نمایش داده نمی‌شود.

| ID | عنوان در داشبورد | دادهٔ منبع | چه چیزی نشان می‌دهد | تفسیر درست |
|---|---|---|---|---|
| G01 | پوشش رکوردهای PCAP به تفکیک پروتکل، split و نوع خروجی | `catalog/artifacts.parquet` | حجم رکوردهای benign/attack و نوع artifact برای هر protocol | برای دیدن پوشش و نامتوازنی منبع است، نه دقت مدل. |
| G02 | تنوع artifactهای تولیدشده در هر پروتکل | catalog | شمار فایل‌های هر kind مانند feature، mapping، quality | نبود یک kind یعنی مرحلهٔ متناظر هنوز کامل نشده یا تولید نشده است. |
| G03 | سهم هر پروتکل از حجم رکوردهای ثبت‌شده | catalog | سهم نسبی recordهای ذخیره‌شده | سهم بیشتر به‌معنی اهمیت یا کیفیت بیشتر protocol نیست. |
| G04 | تعداد Capture در برابر حجم رکورد هر پروتکل/split | catalog | حجم و تعداد فایل‌های ورودی به تفکیک protocol/split | Capture زیاد با record کم می‌تواند محدودیت packet یا ترافیک کم باشد. |
| G05 | حجم ترافیک نمونه در زمان؛ مقایسهٔ همهٔ پروتکل‌ها | feature records نمونه‌ای | تعداد packet نمونه در bucket زمانی برای هر protocol | زمان‌های دارای capture متفاوت را مقایسه می‌کند؛ throughput واقعی کل داده نیست مگر نمونه کل داده باشد. |
| G06 | تغییر زمانی نرمال در برابر حملهٔ برچسب‌خورده | benign و labelled attack samples | تراکم زمانی گروه داده | هم‌پوشانی زمانی، نیاز به split زمانی/بازبینی leakage را یادآوری می‌کند. |
| G07 | تعداد Flow یکتا در هر بازهٔ زمانی | samples | تعداد `flow_id` یکتا به‌ازای protocol/bucket | افزایش می‌تواند اسکن، burst، یا صرفاً workload عادی باشد. |
| G08 | بازهٔ زمانی مشاهده‌شده برای هر پروتکل؛ شروع تا پایان | samples | اولین و آخرین timestamp هر protocol | برای یافتن شکاف زمانی و تفاوت دورهٔ داده‌ها است. |
| G09 | توزیع طول packet در پروتکل‌ها و گروه داده | `packet_length` | violin/box طول packet در نرمال و حمله | اختلاف توزیع یک سرنخ است؛ به‌تنهایی rule تشخیص حمله نیست. |
| G10 | توزیع آنتروپی payload؛ شاخص ساختار/تصادفی‌بودن محتوا | `payload_entropy` | آنتروپی شانون payload | آنتروپی بالا می‌تواند encryption یا دادهٔ فشردهٔ سالم هم باشد. |
| G11 | نرخ packet در Flowها؛ مقایسهٔ فشار ترافیک | `packet_rate` | distribution نرخ packet/s در flow | برای burst/flood سرنخ می‌دهد، نه اثبات DDoS. |
| G12 | طول عمر Flowها در پروتکل‌های مختلف | `flow_duration` | distribution طول flow | کوتاه‌بودن flow می‌تواند رفتار عادی پروتکل باشد. |
| G13 | نرخ ارسال مبدأ؛ نشانهٔ burst و اسکن/سیل احتمالی | `source_packet_rate` | distribution نرخ ارسال source | در نسخهٔ فعلی group سراسری capture است، نه rolling 60-second واقعی. |
| G14 | رابطهٔ طول packet و آنتروپی payload؛ الگوی چندمتغیره | `packet_length`, `payload_entropy`, `burstiness` | clusterها/outlierهای دو ویژگی به‌همراه اندازهٔ burstiness | ارزش آن کشف الگوی ترکیبی است، نه threshold تک‌ستونی. |
| G15 | سهم TCP/UDP در هر پروتکل | `transport` | تعداد packetها به تفکیک transport | پروتکل/پورت تشخیص‌داده‌شده را با transport قاطی نکنید. |
| G16 | توازن دو جهت نرمال‌شدهٔ Flow | `direction` | تعداد packet در endpoint مرتب‌شدهٔ 0/1 | direction نرمال‌شده client/server نیست. |
| G17 | حرکت packet به‌سوی endpoint سرویس | `is_request_direction` | packetهای رو به پورت سرویس در برابر جهت دیگر | proxy/NAT و پورت غیرمتعارف می‌توانند تفسیر را تغییر دهند. |
| G18 | پذیرش/رد نگاشت CSV به PCAP در هر پروتکل | `reports/mapping-audit.parquet` | تعداد captureهای accepted و نیازمند بررسی | فقط accepted برای آموزش Stage 2 قابل اعتمادند. |
| G19 | کیفیت هم‌زمان نگاشت: match rate در برابر confidence | mapping audit | هر capture در صفحهٔ match-rate/confidence | نقطهٔ پایین/چپ باید بررسی دستی شود؛ filename score جای evidence شبکه را نمی‌گیرد. |
| G20 | توزیع offset زمانی جبران‌شده بین CSV و PCAP | mapping audit | offset انتخاب‌شده در mapping | offset یک اصلاح زمان است، نه نشانهٔ حمله. |
| G21 | توازن نوع حمله به‌ازای پروتکل و پذیرش نگاشت | `attack-label-distribution.parquet` | شمار PCAP-derived records برای classها | class کوچک، F1 آن class را ناپایدار می‌کند؛ نمودار برای تصمیم balancing است. |
| G22 | heatmap پوشش featureهای مشترک در پروتکل‌ها | `reports/features/*-overview.parquet` | availability 24 feature پرتکرار در protocolها | صفر/NaN در protocol نامربوط ممکن است طبیعی باشد. |
| G23 | heatmap صفر/تنک‌بودن همان featureها | feature overview | `zero_ratio` همان feature/protocolها | zero زیاد لزوماً بد نیست؛ باید با معنی protocol بررسی شود. |
| G24 | پوشش در برابر تغییرپذیری featureها؛ معیار انتخاب برای مدل | feature overview | availability در برابر IQR و تعداد مقدار یکتا | feature خوب معمولاً هم قابل مشاهده و هم متغیر است؛ هم‌بستگی/redundancy نیز مهم است. |
| G25 | وضعیت استخراج featureها در هر پروتکل | `*-quality.parquet` | تعداد statusهای استخراج/قابلیت مدل | schema placeholder را با feature استخراج‌شده اشتباه نگیرید. |
| G26 | مشاهده‌پذیری و تنوع featureها؛ علامت شکل = قابلیت استفادهٔ مدل | feature quality | observed ratio در برابر unique count | یک feature با unique زیاد ممکن است high-cardinality و گران باشد. |
| G27 | مقایسهٔ معیارهای Stage 1 و Stage 2 میان مدل‌های موجود | `models/**/training-summary.json` | FPR/Recall/F1های مدل‌های آموزش‌دیده | فقط مدل‌های موجود نمایش می‌یابند؛ Stage 1 و Stage 2 مسئلهٔ یکسانی را حل نمی‌کنند. |
| G28 | هزینهٔ آموزش در برابر ابعاد ورودی و حجم ارزیابی | training summary | زمان در برابر transformed inputs و evaluation rows | برای ظرفیت‌سنجی CPU است، benchmark بین ماشین‌های متفاوت نیست. |
| G29 | مصرف RAM پردازش قبل و بعد از آموزش مدل‌ها | `reports/resource-usage.parquet` | RSS process در دو نقطهٔ پایدار | پیک لحظه‌ای RAM را نشان نمی‌دهد؛ فقط before/after ثبت‌شده است. |

## تابلوی کل سامانه: جدول‌ها

| ID | جدول | منبع | کاربرد |
|---|---|---|---|
| T01 | خلاصهٔ پروتکل‌ها | catalog | تعداد فایل، record و بیشترین ستون هر protocol؛ اولین شاخص سلامت coverage. |
| T02 | protocol/split/kind coverage | catalog | دقیقاً مشخص می‌کند کدام split و artifact برای کدام protocol موجود است. |
| T03 | بازهٔ زمانی مشاهده‌شده | feature samples | شروع/پایان UTC، record، Flow و Capture هر protocol. |
| T04 | bucketهای زمانی | feature samples | دادهٔ عددی پشت G05؛ مناسب export یا بررسی spikeها در سیستم دیگر. |
| T05 | پروفایل packet و flow | feature samples | median طول packet، entropy، rate و duration برای protocol/group. |
| T06 | پروفایل Capture | feature samples | حجم، Flow، مبدأ و مقصد هر capture؛ برای یافتن capture غالب. |
| T07 | ممیزی نگاشت | `mapping-audit.parquet` | selected CSV، نرخ تطبیق، confidence، offset و accepted هر capture. |
| T08 | توازن برچسب‌ها | `attack-label-distribution.parquet` | حجم classهای Stage 2 به‌همراه وضعیت acceptance mapping. |
| T09 | featureهای مشاهده‌پذیر/متغیر | feature overview | availability، IQR، std و unique values؛ نامزدهای profile. |
| T10 | featureهای sparse | feature overview | zero ratio بالا و availability پایین؛ علت فنی پرهیز از انتخاب feature. |
| T11 | کیفیت feature | feature quality | status، observed ratio، reason و model usability برای audit. |
| T12 | scorecard مدل | training summary | FPR/Recall/F1، تعداد input، زمان و مسیر هر model artifact. |

## صفحات تک‌پروتکل

وقتی از سایدبار یک protocol انتخاب شود، همهٔ صفحه‌های زیر همان protocol را filter می‌کنند. این‌ها جایگزین تابلوی کل نیستند؛ علت‌یابی دقیق‌تر هستند.

| بخش | visualها و جدول‌ها | سؤال پاسخ‌داده‌شده |
|---|---|---|
| نمای کلی | کارت‌های artifact/feature/label/mapping، جدول راهنمای اجرا، نمودار نوع artifact، نمودار coverage record، نمودارهای وضعیت mapping | آیا این protocol از ورودی تا خروجی مسیر کامل دارد؟ |
| داده و نگاشت | نرخ match هر capture، scatter match-rate/confidence، violin confidence، histogram offset، روش pairing، filename-pairing score، توازن label، جزئیات Flow و جدول ممیزی | CSV واقعاً به دادهٔ PCAP درست وصل شده است؟ |
| فیچرها | availability bar، coverage-vs-IQR scatter، zero-ratio bar، مقایسهٔ نرمال/حمله، histogram/box/ECDF feature انتخابی، scatter چندمتغیره، correlation heatmap، quality table | چه featureهایی قابل مشاهده، متغیر و جداکننده‌اند؟ |
| پروفایل و پیش‌پردازش | Sankey مسیر داده/مدل، pie category، bar هزینهٔ استخراج، جدول تعریف profile، coverage feature انتخابی، barهای raw/prepared IQR و std، جدول comparison | پروفایل دقیقاً چه چیزی را وارد مدل می‌کند و scaling چه اثری داشته است؟ |
| مدل‌ها – Stage 1 | LSTM/Isolation scatter، histogram score، confusion matrix، ROC، Precision–Recall، sensitivity threshold، reconstruction/ensemble distributions، loss curve | گیت anomaly چه trade-offی میان FPR و Recall دارد؟ |
| مدل‌ها – شواهد feature | importance aggregate، coverage-vs-severity، heatmap feature×packet، scatter rank، جدول packet context، bar/polar featureهای یک packet، box comparison و directory anomaly | برای هر anomaly کدام featureها از baseline نرمال دور بوده‌اند؟ |
| مدل‌ها – Stage 2 | feature importance Random Forest، confusion matrix، correct/incorrect per class، sunburst خطا و جدول اشتباه‌ها | نوع حمله پس از عبور از Stage 1 چقدر درست تشخیص داده شده است؟ |
| مدل‌ها – پایپ‌لاین/منابع | flow Stage1→Stage2، donut نوع حمله، prediction table، before/after RAM، ONNX table | آیا pipeline کامل و artifact استقرار آماده است؟ |
| فایل‌ها | catalog خروجی‌ها و download | هر visual و data artifact در کدام مسیر قابل استفادهٔ بیرونی است؟ |

## مانیتورهای زنده

دو مانیتور با polling سه‌ثانیه‌ای وجود دارد؛ هر دو همان فایل‌های پایدار را نشان می‌دهند که پس از بستن dashboard نیز باقی می‌مانند.

| مانیتور | فایل پایدار | ستون‌های جدول رویداد | معنی |
|---|---|---|---|
| استخراج و نگاشت | `<run>/dataset-progress.json` | زمان، مرحله، رویداد، protocol، capture، packet/rows، پیشرفت | نشان می‌دهد کدام capture در حال خواندن، استخراج، quality analysis یا mapping CSV است. |
| آموزش دو مرحله‌ای | `<model>/training-progress.json` | زمان، Stage، epoch، train/validation loss، پیشرفت | وضعیت preprocessing، LSTM epochها، Isolation Forest، calibration، Random Forest، ONNX و پایان را نشان می‌دهد. |
| tail لاگ | `dashboard-dataset-run.log` یا `dashboard-training.log` | آخرین 24 خط CLI | همان log عملیاتی قابل مشاهده در کنسول است؛ برای خطای کامل به فایل log مراجعه کنید. |

## استفاده خارج از dashboard

هر جدول dashboard قابل دانلود CSV است، اما برای integration بهتر است Parquet/JSON اصلی خوانده شود. مسیر واقعی را از «فایل‌ها» یا catalog بگیرید. اگر نموداری را در سامانهٔ دیگری بازسازی می‌کنید، همان filterهای `protocol`، `split`، `mapping_accepted` و UTC bucket را حفظ کنید تا نتیجه مخلوط یا گمراه‌کننده نشود.

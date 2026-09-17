/**
 * Centralized bilingual (Arabic/English) translation system for the
 * SignBridge interface. This controls only the STATIC interface text
 * (labels, headings, statuses, buttons, aria-labels) — it never touches
 * what the user types, what the backend returns as an educational answer,
 * or any RAG/recognition logic.
 *
 * Usage:
 *   import { t, getLanguage, setLanguage, applyTranslations, onLanguageChange } from './i18n.js';
 *   t('hero.headline')                         -> plain string lookup
 *   t('experience.avatar.playing', { token })  -> {token}-style interpolation
 *
 * `applyTranslations()` walks every element carrying data-i18n(-*) and
 * fills it in from the dictionary for the current language — this is the
 * single place that keeps HTML and dictionary in sync, instead of
 * scattering per-element conditionals through the codebase.
 */

const STORAGE_KEY = 'signbridge-language';

const DICTIONARY = {
  ar: {
    'meta.title': 'SignBridge | سِجن بريدج',
    'meta.description': 'سِجن بريدج: مساعد تعليمي بلغة الإشارة الأردنية لمواد علوم الحاسوب الأساسية',

    'a11y.skip_link': 'تخطَّ إلى المحتوى الرئيسي',
    'a11y.header_logo': 'شعار SignBridge',
    'a11y.hero_logo': 'شعار SignBridge: يد ترفع إشارة تواصل بجانب يد ثانية',
    'a11y.header_nav': 'روابط رئيسية',
    'a11y.mode_switch': 'طريقة الإدخال',
    'a11y.view_switch': 'اختيار زاوية العرض',
    'a11y.avatar_stage': 'عرض الأفاتار ثلاثي الأبعاد',
    'a11y.brand_home': 'SignBridge — العودة إلى الصفحة الرئيسية',
    'a11y.mobile_menu_open': 'فتح القائمة',
    'a11y.mobile_menu_close': 'إغلاق القائمة',
    'a11y.lang_switch': 'اختيار لغة الواجهة',
    'a11y.theme_switch': 'تبديل المظهر الفاتح والداكن',
    'a11y.color_vision_switch': 'تبديل لوحة ألوان مناسبة لعمى الألوان',
    'a11y.features_explorer': 'استكشاف الإمكانيات',
    'a11y.resources_explorer': 'استكشاف مصادر البيانات',
    'a11y.dev_toggle': 'خيارات متقدمة للمطورين',
    'a11y.recording_indicator': 'جارٍ التسجيل',

    'nav.about': 'حول المنصة',
    'nav.how': 'كيف تعمل',
    'nav.features': 'الإمكانيات',
    'nav.resources': 'مصادر البيانات',
    'nav.experience': 'جرّب المنصة',
    'nav.cta': 'جرّب SignBridge',
    'lang.ar': 'العربية',
    'lang.en': 'English',
    'theme.light': 'فاتح',
    'theme.dark': 'داكن',
    'accessibility.colorblind_off': 'ألوان ميسّرة',
    'accessibility.colorblind_on': 'ألوان ميسّرة ✓',

    'hero.headline': 'اسأل. أشر. افهم.',
    'hero.description': 'يفهم SignBridge أسئلتك المكتوبة أو بلغة الإشارة الأردنية.',
    'home.note': 'أردن أكثر شمولًا يبدأ بالفهم.',
    'hero.cta.primary': 'جرّب المنصة الآن',
    'hero.cta.secondary': 'شاهد الأفاتار',

    'project.eyebrow': 'منصة تعليمية بلغة الإشارة الأردنية',
    'project.title': 'تعلّم أوضح، أبسط، وأقرب للجميع.',
    'project.lead': 'SignBridge يقدّم تجربة تعليمية ناعمة وسهلة لطلبة الصم وضعاف السمع، من خلال الأسئلة النصية، وفهم لغة الإشارة الأردنية، وعرض الإجابة بصيغة بصرية واضحة.',
    'project.learn_more': 'تعرّف على المشروع',
    'project.university_title': 'طُوّر بواسطة طلبة من الجامعة الأردنية',
    'project.university_subtitle': 'مشروع طلابي في الذكاء الاصطناعي من أجل تعليم أكثر شمولًا.',
    'project.team_heading': 'من نحن',
    'project.team_body': 'نحن طلبة من الجامعة الأردنية نطوّر SignBridge كمشروع تقني تعليمي مبني على البحث، ويركّز على الإتاحة والشمول وتطبيقات الذكاء الاصطناعي العملية.',
    'project.accessibility_heading': 'مصمم للوصول المرن',
    'project.accessibility_body': 'بدّل بين العربية والإنجليزية، والمظهر الفاتح والداكن، ولوحة ألوان مناسبة لعمى الألوان من أعلى الصفحة.',

    'about.eyebrow': 'حول المنصة',
    'about.title': 'مساعد تعليمي مصمَّم لطلبة الصم وضعاف السمع',
    'about.body':
      'يجمع SignBridge بين الإجابات التعليمية المدعومة بالذكاء الاصطناعي والتعرف على لغة الإشارة، ليتيح للطلبة الصم وضعاف السمع الوصول إلى المواد الدراسية بطريقة مباشرة ومفهومة.',
    'about.point1.title': 'إجابات موثّقة بالمصادر',
    'about.point1.body': 'كل إجابة تعليمية مبنية على مادة دراسية فعلية، مع ذكر المصادر التي استُخرجت منها.',
    'about.point2.title': 'تعرّف تلقائي على الإشارة',
    'about.point2.body': 'يحدد النظام تلقائيًا الأسلوب الأنسب لتحليل الإشارة دون الحاجة لاختيار يدوي.',
    'about.point3.title': 'عرض بصري بلغة الإشارة',
    'about.point3.body': 'أفاتار ثلاثي الأبعاد يعرض ما هو متاح من حركات الإشارة المرتبطة بالإجابة.',

    'how.eyebrow': 'كيف تعمل المنصة',
    'how.title': 'من السؤال إلى الإجابة المرئية',
    'how.step1.title': 'اطرح سؤالك أو قدّم إشارة',
    'how.step1.body': 'اكتب سؤالًا دراسيًا، أو سجّل إشارة عبر الكاميرا، أو ارفع مقطع فيديو.',
    'how.step2.title': 'يعالج SignBridge الطلب',
    'how.step2.body': 'يوجّه النظام السؤال أو الإشارة تلقائيًا إلى المادة التعليمية ذات الصلة ويسترجع المعلومات المناسبة.',
    'how.step3.title': 'إجابة نصية موثّقة',
    'how.step3.body': 'تظهر إجابة تعليمية واضحة مع نسخة مبسطة، وقائمة بالمصادر التي استُندت إليها الإجابة.',
    'how.step4.title': 'عرض بلغة الإشارة عند التوفر',
    'how.step4.body': 'إذا كانت حركات الإشارة المرتبطة بالإجابة متوفرة، يعرضها الأفاتار الثلاثي الأبعاد.',

    'features.eyebrow': 'الإمكانيات الحالية',
    'features.title': 'ما يقدّمه SignBridge اليوم',
    'features.item1.title': 'إجابة أسئلة تعليمية',
    'features.item1.body': 'اسأل عن مفاهيم أساسية في علوم الحاسوب واحصل على شرح مباشر.',
    'features.item2.title': 'إجابات مدعومة بالاسترجاع (RAG)',
    'features.item2.body': 'كل إجابة مبنية على مقاطع فعلية من المادة الدراسية، مع ذكر مصدرها.',
    'features.item3.title': 'دعم العربية والإنجليزية',
    'features.item3.body': 'اطرح سؤالك بأي من اللغتين، بشكل مستقل عن لغة واجهة الموقع.',
    'features.item4.title': 'تعرّف عبر الكاميرا',
    'features.item4.body': 'سجّلي إشارة مباشرة من الكاميرا لتحليلها.',
    'features.item5.title': 'تحليل فيديو مرفوع',
    'features.item5.body': 'ارفع مقطع فيديو يحتوي إشارة أو أكثر لتحليله.',
    'features.item6.title': 'توجيه تلقائي للتعرف',
    'features.item6.body': 'يختار النظام تلقائيًا الأسلوب المناسب لتحليل الإشارة بدون تدخل يدوي.',
    'features.item7.title': 'مكتبة حركات',
    'features.item7.body': 'تصفّح وتشغيل رموز الحركة المتوفرة عبر مجموعات البيانات الثلاث.',
    'features.item8.title': 'عرض أفاتار ثلاثي الأبعاد',
    'features.item8.body': 'تقديم بصري لحركات لغة الإشارة عبر أفاتار تفاعلي.',

    'resources.eyebrow': 'مصادر لغة الإشارة',
    'resources.title': 'مبني على ثلاث مجموعات بيانات للإشارة',
    'resources.jordanian.title': 'الإشارات الأردنية التقنية',
    'resources.jordanian.body': 'مجموعة إشارات معزولة لمصطلحات تقنية في علوم الحاسوب.',
    'resources.karsl.title': 'KArSL',
    'resources.karsl.body': 'مجموعة بيانات عربية معروفة للإشارات المعزولة.',
    'resources.isharah.title': 'Isharah',
    'resources.isharah.body': 'مجموعة بيانات للإشارات المتصلة (جمل متعددة الإشارات).',

    'experience.eyebrow': 'التجربة التفاعلية',
    'experience.title': 'جرّب SignBridge الآن',
    'experience.description': 'اسأل سؤالًا، أو استخدم الكاميرا، أو ارفع فيديو — والأفاتار جاهز لعرض الإشارة المتاحة.',
    'thanks.eyebrow': 'شكر وامتنان',
    'thanks.title': 'كل الشكر لأحمد وماجد',
    'thanks.body': 'نتقدّم بجزيل الشكر والامتنان إلى أحمد وماجد على دعمهما الكريم ومساهمتهما القيّمة في جمع البيانات. كان لتعاونهما أثر مهم في تطوير SignBridge والوصول به إلى هذه المرحلة.',
    'thanks.note': 'شكراً لوقتكما وثقتكما ومساندتكما الصادقة.',
    'thanks.video_title': 'فيديو الشكر',
    'thanks.video_body': 'لقطة قصيرة توثّق جزءًا من هذا التعاون الجميل.',
    'thanks.illustration_title': 'أحمد وماجد',
    'thanks.illustration_body': 'تقدير بسيط لشابين كان لهما دور جميل ومؤثر في دعم المشروع.',
    'thanks.illustration_alt': 'رسمة تقديرية بأسلوب أنمي لشابين ساهما في جمع البيانات لمشروع SignBridge',

    'backend.connecting': 'جارٍ الاتصال بالخادم…',
    'backend.connected': 'الخادم متصل',
    'backend.connected_model': 'الخادم متصل (النموذج محمّل)',
    'backend.error': 'تعذر الاتصال بالخادم',

    'avatar.loading': 'جارٍ تجهيز الأفاتار…',
    'avatar.ready': 'جاهز',
    'avatar.error_title': 'تعذر تحميل الأفاتار',
    'avatar.error_message': 'تعذر تحميل الأفاتار الثلاثي الأبعاد. النص التعليمي سيبقى متاحًا.',
    'avatar.view.signing': 'وضع الإشارة',
    'avatar.view.full': 'الجسم كاملًا',
    'avatar.now_playing.none': 'لا توجد حركة قيد التشغيل',
    'avatar.now_playing.playing': 'قيد التشغيل: {token}',
    'avatar.now_playing.skip': 'تخطي (لا توجد حركة): {token}',
    'avatar.now_playing.complete': 'اكتمل التشغيل',
    'avatar.now_playing.stopped': 'تم الإيقاف',
    'avatar.replay': 'إعادة التشغيل',
    'avatar.stop': 'إيقاف',

    'tools.eyebrow': 'أدوات إضافية',
    'tools.title': 'الكاميرا، الفيديو، ومكتبة الحركات',

    'flow.entry.eyebrow': 'ابدأ من هنا',
    'flow.entry.title': 'كيف تودّ أن تسأل؟',
    'flow.entry.lead': 'اختَر الطريقة التي تناسبك أكثر.',
    'flow.choice.ask.title': 'اكتب سؤالك',
    'flow.choice.ask.hint': 'اكتب سؤالك نصًا بالعربية أو الإنجليزية',
    'flow.choice.camera.title': 'سجّل فيديو بالإشارة',
    'flow.choice.camera.hint': 'استخدم الكاميرا مباشرة',
    'flow.choice.upload.title': 'ارفع فيديو',
    'flow.choice.upload.hint': 'اختر مقطعًا جاهزًا من جهازك',
    'flow.choice.library.title': 'مكتبة الحركات',
    'flow.choice.library.hint': 'تصفّح الإشارات الجاهزة وشغّلها مباشرة',
    'flow.back': 'رجوع',

    'tabs.ask': 'اسأل سؤالاً',
    'tabs.camera': 'استخدم الكاميرا',
    'tabs.upload': 'ارفع فيديو',
    'tabs.library': 'تصفّح مكتبة الحركات',
    'library.featured': 'استكشف المزيد',
    'library.spotlight': 'ابدئي من هنا لتصفّح الحركات الجاهزة بسهولة.',

    'ask.title': 'ماذا تود أن تسأل؟',
    'ask.help': 'اكتب سؤالك بالعربية أو الإنجليزية، وسيساعدك SignBridge.',
    'ask.label': 'سؤالك',
    'ask.placeholder': 'اكتب سؤالك بالعربية أو الإنجليزية…',
    'ask.error': 'الرجاء كتابة سؤال قبل الإرسال.',
    'ask.submit': 'اسأل SignBridge',
    'ask.alt_prefix': 'تفضّل الإشارة؟',
    'ask.alt_help': 'يمكنك طرح سؤالك بلغة الإشارة الأردنية بدلًا من ذلك.',
    'question.suggested': 'جرّب واحدًا من هذه',
    'question.suggest1': 'اشرح مفهومًا',
    'question.suggest2': 'اسأل عن تقنية',
    'question.suggest3': 'ترجم سؤالي',
    'ask.loading': 'جارٍ البحث عن إجابة…',
    'ask.generic_error': 'حدث خطأ غير متوقع.',

    'camera.title': 'تسجيل إشارة بالكاميرا',
    'camera.help': 'اضغط "تشغيل الكاميرا" أولاً، ثم سجّل الإشارة وأوقف التسجيل قبل التحليل.',
    'camera.page_title': 'سجّل سؤالك',
    'camera.page_help': 'أنشئ فيديو قصيرًا وواضحًا بالإشارة، واترك SignBridge يفهمه.',
    'camera.placeholder': 'الكاميرا جاهزة',
    'camera.tips_title': 'لأفضل نتيجة',
    'camera.tips_lead': 'اتّبع هذه النصائح البسيطة قبل التسجيل.',
    'camera.tip1_title': 'أبقِ يديك ظاهرتين',
    'camera.tip1_body': 'أظهر الجزء العلوي من جسمك ويديك داخل الإطار.',
    'camera.tip2_title': 'استخدم إضاءة جيدة',
    'camera.tip2_body': 'واجِه مصدر ضوء وتجنّب الخلفيات الداكنة.',
    'camera.tip3_title': 'اسأل سؤالًا واحدًا كل مرة',
    'camera.tip3_body': 'اجعل سؤالك واضحًا وبسيطًا.',
    'camera.start': 'تشغيل الكاميرا',
    'camera.record': 'ابدأ التسجيل',
    'camera.stop_record': 'إيقاف التسجيل',
    'camera.retake': 'إعادة التسجيل',
    'camera.analyze': 'تحليل الإشارة',
    'camera.close': 'إغلاق الكاميرا',
    'camera.status.requesting': 'جارٍ طلب إذن الكاميرا…',
    'camera.status.ready': 'الكاميرا جاهزة. اضغط "ابدأ التسجيل".',
    'camera.status.recording': 'جارٍ التسجيل…',
    'camera.status.recorded': 'تم التسجيل. يمكنك إعادة التسجيل أو تحليل الإشارة.',
    'camera.status.retake_ready': 'جاهزة لتسجيل جديد.',
    'camera.status.closed': 'تم إغلاق الكاميرا.',
    'camera.status.uploading': 'جارٍ رفع الفيديو وتحليله…',
    'camera.status.completed': 'اكتمل التعرف على الإشارة.',
    'camera.status.failed': 'تعذر التعرف على الإشارة.',
    'camera.status.generic_error': 'تعذر تحليل الفيديو.',
    'camera.error.permission_denied': 'تم رفض إذن الوصول إلى الكاميرا. فعّل الإذن من إعدادات المتصفح وحاول مرة أخرى.',
    'camera.error.not_found': 'لم يتم العثور على كاميرا متصلة بهذا الجهاز.',
    'camera.error.in_use': 'الكاميرا قيد الاستخدام حاليًا من قبل تطبيق آخر.',
    'camera.error.overconstrained': 'تعذر تشغيل الكاميرا بالإعدادات المطلوبة.',
    'camera.error.insecure_context': 'الوصول إلى الكاميرا غير مسموح في هذا السياق (يتطلب اتصالاً آمنًا).',
    'camera.error.unsupported': 'هذا المتصفح لا يدعم تسجيل الفيديو من الكاميرا.',
    'camera.error.not_started': 'لم يتم تشغيل الكاميرا بعد.',
    'camera.error.generic': 'تعذر الوصول إلى الكاميرا.',

    'upload.title': 'رفع فيديو إشارة',
    'upload.help': 'اختر ملف فيديو يحتوي إشارة واحدة أو أكثر لتحليلها تلقائيًا.',
    'upload.label': 'ملف الفيديو (mp4, mov, webm, avi, mkv)',
    'upload.submit': 'تحليل الفيديو',
    'upload.choose_another': 'اختر فيديو آخر',

    'library.title': 'مكتبة حركات الأفاتار',
    'library.help': 'تصفّح جميع رموز الحركة عبر مجموعات البيانات الثلاث، وابحث بالعربية أو الإنجليزية، وشغّل أي حركة جاهزة مباشرة.',
    'library.search_placeholder': 'ابحث بالعربية أو الإنجليزية (مثال: Stack أو مكدس)',
    'library.filter.dataset': 'تصفية حسب المجموعة',
    'library.filter.status': 'تصفية حسب الحالة',
    'library.filter.all': 'الكل',
    'library.filter.jordanian_it': 'الإشارات الأردنية',
    'library.filter.karsl': 'KArSL',
    'library.filter.isharah': 'Isharah',
    'library.status.all': 'كل الحالات',
    'library.status.ready': 'جاهزة فقط',
    'library.status.source_unavailable': 'غير المتوفرة',
    'library.dataset.jordanian_it': 'الإشارات الأردنية التقنية',
    'library.dataset.karsl': 'KArSL',
    'library.dataset.isharah': 'Isharah',
    'library.status_label.ready': 'جاهزة',
    'library.status_label.pending_generation': 'قيد الإعداد',
    'library.status_label.source_unavailable': 'غير متوفرة',
    'library.status_label.incompatible_source': 'مصدر غير متوافق',
    'library.status_label.extraction_failed': 'فشل الاستخراج',
    'library.status_label.invalid_motion': 'ملف غير صالح',
    'library.status_label.validation_required': 'بانتظار التحقق',
    'library.loading': 'جارٍ تحميل قائمة الحركات…',
    'library.load_error': 'تعذر تحميل مكتبة الحركات. تأكدي من تشغيل الخادم.',
    'library.summary_capped': 'عرض أول {shown} من {matched} نتيجة مطابقة (الإجمالي {total} رمزًا، {ready} جاهزًا) — استخدم البحث لتضييق النتائج.',
    'library.summary': '{matched} نتيجة (الإجمالي {total} رمزًا، {ready} جاهزًا).',
    'library.unavailable_note_title': 'الإشارات الست غير المتوفرة في Isharah:',
    'library.empty': 'لا توجد نتائج مطابقة.',
    'library.play': 'تشغيل',
    'library.playing': 'قيد التشغيل…',

    'results.recognition.title': 'نتيجة التعرف على الإشارة',
    'results.recognition.dev_details': 'تفاصيل تشخيصية (للمطورين)',
    'results.recognition.requested_mode': 'الوضع المطلوب',
    'results.recognition.resolved_mode': 'الوضع المستخدم فعليًا',
    'results.recognition.accepted': 'مقبول؟',
    'results.recognition.yes': 'نعم',
    'results.recognition.no': 'لا',
    'results.recognition.reason': 'سبب القرار',
    'results.recognition.predicted_label': 'التسمية المتوقعة',
    'results.recognition.confidence': 'نسبة الثقة',
    'results.recognition.frame_count': 'عدد الإطارات',
    'results.recognition.active_segment': 'المقطع النشط',
    'results.recognition.top_k': 'أفضل الاحتمالات',
    'results.recognition.candidate_scores': 'درجات المسارات المرشحة',
    'results.recognition.motion_label': 'حركة الإشارة المتعرف عليها',
    'results.recognition.motion_available': 'متوفرة ({tokens})',
    'results.recognition.motion_unavailable': 'غير متوفرة حاليًا ({tokens})',
    'results.recognition.retry_message':
      'تعذّر التعرف على الإشارة بثقة كافية. الرجاء إعادة التسجيل مع التأكد من: ظهور الجزء العلوي من الجسم واليدين بوضوح، ثبات الكاميرا، إضاءة كافية، وتسجيل الإشارة كاملة.',
    'results.recognition.recognized': 'تم التعرف على: {label}',
    'results.recognition.recognized_confidence': 'تم التعرف على: {label} (بثقة {percent}%)',
    'results.recognition.recognized_confidence_avg': 'تم التعرف على: {label} (متوسط ثقة الفك: {percent}%)',
    'results.recognition.none': 'لم يتم التعرف على أي إشارة في هذا المقطع.',
    'results.recognition.na': '—',

    'results.page_title': 'سؤالك',
    'results.page_help': 'هذا ما فهمناه والإجابة بالنص ولغة الإشارة.',
    'results.your_question': 'السؤال المكتشف: {question}',
    'results.ask_again': 'اسأل سؤالاً آخر',

    'results.answer.title': 'الإجابة',
    'results.answer.simplified_title': 'مثال',
    'results.answer.topic': 'الموضوع: {topics}',
    'results.answer.unavailable_default': 'لا تتوفر إجابة تعليمية لهذا الإدخال.',
    'results.answer.no_gloss': 'لم يُنتج نظام التعرف رمزًا قابلًا للاستخدام لهذا المقطع.',
    'results.answer.details_summary': 'المصادر وتفاصيل إضافية',

    'results.sources.title': 'المصادر',
    'results.sources.item': '{domain} — {title} (دقة الاسترجاع {score})',

    'results.tokens.title': 'سلسلة رموز الإشارة',
    'results.tokens.disclaimer': 'هذه سلسلة مفاهيم لتخطيط الأفاتار، وليست نحوًا موثّقًا بالكامل للغة الإشارة الأردنية.',

    'results.motions.title': 'حالة حركات الأفاتار',
    'results.motions.coverage': 'الحركات المتاحة: {percent}% من رموز الإشارة',
    'results.motions.missing': 'حركات غير متوفرة حاليًا: {tokens}',

    'devmode.summary': 'خيارات متقدمة (للمطورين فقط)',
    'devmode.auto': 'تلقائي (الوضع الافتراضي)',
    'devmode.it': 'فرض: الإشارات الأردنية التقنية',
    'devmode.karsl': 'فرض: KArSL',
    'devmode.continuous': 'فرض: إشارات متصلة',

    'final_cta.title': 'جاهز لتجربة SignBridge؟',
    'final_cta.description': 'اسأل سؤالك الأول، أو سجّل إشارة، وشاهد الأفاتار يعرضها الآن.',
    'final_cta.button': 'الانتقال إلى التجربة التفاعلية',

    'footer.tagline': 'مساعد تعليمي بلغة الإشارة الأردنية، مدعوم بالذكاء الاصطناعي.',
    'footer.nav_title': 'الأقسام',
    'footer.resources_title': 'مصادر البيانات',
    'footer.built_with': 'مشروع طلابي من الجامعة الأردنية يركّز على التعليم الشامل وإتاحة الوصول.',
    'footer.thanks_link': 'الشكر',
    'footer.contact_title': 'للتواصل',
    'footer.contact_note': 'للتواصل بخصوص المشروع أو التعاون.',
    'footer.credit': 'مع خالص الشكر لأحمد وماجد.',
    'footer.rights_note': 'صُمّم بعناية لتجربة تعليمية أكثر شمولًا.',
    'footer.rights': '© {year} SignBridge',
  },

  en: {
    'meta.title': 'SignBridge | Sign-Language Learning Assistant',
    'meta.description': 'SignBridge: an AI-powered educational assistant with Jordanian Sign Language support for core computer-science material.',

    'a11y.skip_link': 'Skip to main content',
    'a11y.header_logo': 'SignBridge logo',
    'a11y.hero_logo': 'SignBridge logo: a raised hand signaling connection beside a second hand',
    'a11y.header_nav': 'Main navigation',
    'a11y.mode_switch': 'Input method',
    'a11y.view_switch': 'Camera angle',
    'a11y.avatar_stage': '3D avatar viewer',
    'a11y.brand_home': 'SignBridge — return to the homepage',
    'a11y.mobile_menu_open': 'Open menu',
    'a11y.mobile_menu_close': 'Close menu',
    'a11y.lang_switch': 'Choose interface language',
    'a11y.theme_switch': 'Toggle light and dark appearance',
    'a11y.color_vision_switch': 'Toggle a color-vision-friendly palette',
    'a11y.features_explorer': 'Explore capabilities',
    'a11y.resources_explorer': 'Explore data sources',
    'a11y.dev_toggle': 'Advanced developer options',
    'a11y.recording_indicator': 'Recording',

    'nav.about': 'About',
    'nav.how': 'How it works',
    'nav.features': 'Features',
    'nav.resources': 'Data sources',
    'nav.experience': 'Try the platform',
    'nav.cta': 'Try SignBridge',
    'lang.ar': 'العربية',
    'lang.en': 'English',
    'theme.light': 'Light',
    'theme.dark': 'Dark',
    'accessibility.colorblind_off': 'Color-friendly',
    'accessibility.colorblind_on': 'Color-friendly ✓',

    'hero.headline': 'Ask. Sign. Understand.',
    'hero.description': 'SignBridge understands your questions in text or Jordanian Sign Language.',
    'home.note': 'A more inclusive Jordan starts with understanding.',
    'hero.cta.primary': 'Try SignBridge now',
    'hero.cta.secondary': 'Watch the avatar',

    'project.eyebrow': 'A Jordanian Sign Language learning platform',
    'project.title': 'Clearer learning, made simpler and more inclusive.',
    'project.lead': 'SignBridge offers a smooth educational experience for deaf and hard-of-hearing students through text questions, Jordanian Sign Language understanding, and clear visual answers.',
    'project.learn_more': 'Learn about the project',
    'project.university_title': 'Built by University of Jordan students',
    'project.university_subtitle': 'A student-led AI project for more inclusive education.',
    'project.team_heading': 'Who we are',
    'project.team_body': 'We are students from the University of Jordan developing SignBridge as a research-driven educational technology project focused on accessibility, inclusion, and practical AI.',
    'project.accessibility_heading': 'Designed for flexible access',
    'project.accessibility_body': 'Switch between Arabic and English, light and dark appearance, and a color-vision-friendly palette from the header.',

    'about.eyebrow': 'About the platform',
    'about.title': 'A learning assistant built for deaf and hard-of-hearing students',
    'about.body':
      'SignBridge combines AI-powered educational answers with sign-language recognition, giving deaf and hard-of-hearing students direct, understandable access to their course material.',
    'about.point1.title': 'Source-backed answers',
    'about.point1.body': 'Every educational answer is grounded in real course material, with its sources listed.',
    'about.point2.title': 'Automatic sign recognition',
    'about.point2.body': 'The system automatically chooses the right way to analyze a sign — no manual selection needed.',
    'about.point3.title': 'Sign-language visual presentation',
    'about.point3.body': 'A 3D avatar presents whatever sign motion is available for the answer.',

    'how.eyebrow': 'How it works',
    'how.title': 'From question to visual answer',
    'how.step1.title': 'Ask or sign',
    'how.step1.body': 'Type a study question, record a sign with the camera, or upload a video.',
    'how.step2.title': 'SignBridge processes the request',
    'how.step2.body': 'The system automatically routes the question or sign to the relevant material and retrieves the right information.',
    'how.step3.title': 'A source-backed text answer',
    'how.step3.body': 'A clear educational answer appears with a simplified version and the sources it was based on.',
    'how.step4.title': 'Sign-language presentation when available',
    'how.step4.body': 'If sign motions related to the answer are available, the 3D avatar presents them.',

    'features.eyebrow': 'Current capabilities',
    'features.title': 'What SignBridge offers today',
    'features.item1.title': 'Educational Q&A',
    'features.item1.body': 'Ask about core computer-science concepts and get a direct explanation.',
    'features.item2.title': 'Retrieval-backed (RAG) answers',
    'features.item2.body': 'Every answer is grounded in real passages from the course material, with sources cited.',
    'features.item3.title': 'Arabic and English support',
    'features.item3.body': 'Ask your question in either language, independent of the site interface language.',
    'features.item4.title': 'Camera-based recognition',
    'features.item4.body': 'Record a sign directly from the camera for analysis.',
    'features.item5.title': 'Uploaded-video analysis',
    'features.item5.body': 'Upload a video containing one or more signs for analysis.',
    'features.item6.title': 'Automatic recognition routing',
    'features.item6.body': 'The system automatically picks the right recognition path — no manual choice needed.',
    'features.item7.title': 'Motion library',
    'features.item7.body': 'Browse and play the available motion tokens across all three datasets.',
    'features.item8.title': '3D avatar presentation',
    'features.item8.body': 'Visual presentation of sign-language motion through an interactive avatar.',

    'resources.eyebrow': 'Sign-language resources',
    'resources.title': 'Built on three sign-language datasets',
    'resources.jordanian.title': 'Jordanian Technical Signs',
    'resources.jordanian.body': 'An isolated-sign set covering technical computer-science terminology.',
    'resources.karsl.title': 'KArSL',
    'resources.karsl.body': 'A well-known Arabic isolated sign-language dataset.',
    'resources.isharah.title': 'Isharah',
    'resources.isharah.body': 'A continuous sign-language dataset (multi-sign sequences).',

    'experience.eyebrow': 'Interactive experience',
    'experience.title': 'Try SignBridge now',
    'experience.description': 'Ask a question, use the camera, or upload a video — the avatar is ready to present whatever sign is available.',
    'thanks.eyebrow': 'With gratitude',
    'thanks.title': 'Special thanks to Ahmad and Majed',
    'thanks.body': 'We extend our sincere appreciation to Ahmad and Majed for their generous support and valuable contribution to data collection. Their collaboration played an important role in shaping SignBridge.',
    'thanks.note': 'Thank you for your time, trust, and sincere support.',
    'thanks.video_title': 'Thank-you video',
    'thanks.video_body': 'A short video that captures part of this beautiful collaboration.',
    'thanks.illustration_title': 'Ahmad and Majed',
    'thanks.illustration_body': 'A simple tribute to two young men whose support made a meaningful difference.',
    'thanks.illustration_alt': 'Anime tribute illustration for two young men who contributed to data collection for SignBridge',

    'backend.connecting': 'Connecting to server…',
    'backend.connected': 'Server connected',
    'backend.connected_model': 'Server connected (model loaded)',
    'backend.error': 'Could not reach the server',

    'avatar.loading': 'Preparing the avatar…',
    'avatar.ready': 'Ready',
    'avatar.error_title': 'Could not load the avatar',
    'avatar.error_message': 'The 3D avatar could not be loaded. The educational text will still be available.',
    'avatar.view.signing': 'Signing view',
    'avatar.view.full': 'Full body',
    'avatar.now_playing.none': 'No motion currently playing',
    'avatar.now_playing.playing': 'Playing: {token}',
    'avatar.now_playing.skip': 'Skipped (no motion): {token}',
    'avatar.now_playing.complete': 'Playback complete',
    'avatar.now_playing.stopped': 'Stopped',
    'avatar.replay': 'Replay',
    'avatar.stop': 'Stop',

    'tools.eyebrow': 'More tools',
    'tools.title': 'Camera, video upload, and motion library',

    'flow.entry.eyebrow': 'Start here',
    'flow.entry.title': 'How would you like to ask?',
    'flow.entry.lead': 'Choose the way that feels best for you.',
    'flow.choice.ask.title': 'Type your question',
    'flow.choice.ask.hint': 'Write your question in Arabic or English',
    'flow.choice.camera.title': 'Record a sign video',
    'flow.choice.camera.hint': 'Use your camera directly',
    'flow.choice.upload.title': 'Upload a video',
    'flow.choice.upload.hint': 'Choose a clip from your device',
    'flow.choice.library.title': 'Motion library',
    'flow.choice.library.hint': 'Browse ready signs and play them instantly',
    'flow.back': 'Back',

    'tabs.ask': 'Ask a question',
    'tabs.camera': 'Use the camera',
    'tabs.upload': 'Upload a video',
    'tabs.library': 'Browse motion library',
    'library.featured': 'Explore more',
    'library.spotlight': 'Start here to browse ready motions with ease.',

    'ask.title': 'What would you like to ask?',
    'ask.help': 'Type your question in Arabic or English, and SignBridge will help you.',
    'ask.label': 'Your question',
    'ask.placeholder': 'Type your question in Arabic or English…',
    'ask.error': 'Please write a question before submitting.',
    'ask.submit': 'Ask SignBridge',
    'ask.alt_prefix': 'Prefer to sign?',
    'ask.alt_help': 'You can show your question in Jordanian Sign Language instead.',
    'question.suggested': 'Try one of these',
    'question.suggest1': 'Explain a concept',
    'question.suggest2': 'Ask about IT',
    'question.suggest3': 'Translate my question',
    'ask.loading': 'Searching for an answer…',
    'ask.generic_error': 'An unexpected error occurred.',

    'camera.title': 'Record a sign with the camera',
    'camera.help': 'Press "Start camera" first, then record the sign and stop recording before analyzing.',
    'camera.page_title': 'Record your question',
    'camera.page_help': 'Make a short, clear sign video and let SignBridge understand it.',
    'camera.placeholder': 'Camera ready',
    'camera.tips_title': 'For the best result',
    'camera.tips_lead': 'Follow these simple tips before you record.',
    'camera.tip1_title': 'Keep hands visible',
    'camera.tip1_body': 'Show your upper body and hands in the frame.',
    'camera.tip2_title': 'Use good lighting',
    'camera.tip2_body': 'Face a light source and avoid dark backgrounds.',
    'camera.tip3_title': 'Ask one question at a time',
    'camera.tip3_body': 'Keep your question clear and simple.',
    'camera.start': 'Start camera',
    'camera.record': 'Start recording',
    'camera.stop_record': 'Stop recording',
    'camera.retake': 'Retake',
    'camera.analyze': 'Analyze sign',
    'camera.close': 'Close camera',
    'camera.status.requesting': 'Requesting camera permission…',
    'camera.status.ready': 'Camera ready. Press "Start recording".',
    'camera.status.recording': 'Recording…',
    'camera.status.recorded': 'Recorded. You can retake or analyze the sign.',
    'camera.status.retake_ready': 'Ready for a new recording.',
    'camera.status.closed': 'Camera closed.',
    'camera.status.uploading': 'Uploading and analyzing the video…',
    'camera.status.completed': 'Sign recognition complete.',
    'camera.status.failed': 'Could not recognize the sign.',
    'camera.status.generic_error': 'Could not analyze the video.',
    'camera.error.permission_denied': 'Camera access was denied. Enable it in your browser settings and try again.',
    'camera.error.not_found': 'No camera was found on this device.',
    'camera.error.in_use': 'The camera is currently in use by another application.',
    'camera.error.overconstrained': 'Could not start the camera with the requested settings.',
    'camera.error.insecure_context': 'Camera access is not allowed in this context (a secure connection is required).',
    'camera.error.unsupported': 'This browser does not support recording video from a camera.',
    'camera.error.not_started': 'The camera has not been started yet.',
    'camera.error.generic': 'Could not access the camera.',

    'upload.title': 'Upload a sign video',
    'upload.help': 'Choose a video file containing one or more signs to analyze automatically.',
    'upload.label': 'Video file (mp4, mov, webm, avi, mkv)',
    'upload.submit': 'Analyze video',
    'upload.choose_another': 'Choose another video',

    'library.title': 'Avatar motion library',
    'library.help': 'Browse every motion token across all three datasets, search in Arabic or English, and play any ready motion directly.',
    'library.search_placeholder': 'Search in Arabic or English (e.g. Stack or مكدس)',
    'library.filter.dataset': 'Filter by dataset',
    'library.filter.status': 'Filter by status',
    'library.filter.all': 'All',
    'library.filter.jordanian_it': 'Jordanian Signs',
    'library.filter.karsl': 'KArSL',
    'library.filter.isharah': 'Isharah',
    'library.status.all': 'All statuses',
    'library.status.ready': 'Ready only',
    'library.status.source_unavailable': 'Unavailable',
    'library.dataset.jordanian_it': 'Jordanian Technical Signs',
    'library.dataset.karsl': 'KArSL',
    'library.dataset.isharah': 'Isharah',
    'library.status_label.ready': 'Ready',
    'library.status_label.pending_generation': 'In progress',
    'library.status_label.source_unavailable': 'Unavailable',
    'library.status_label.incompatible_source': 'Incompatible source',
    'library.status_label.extraction_failed': 'Extraction failed',
    'library.status_label.invalid_motion': 'Invalid file',
    'library.status_label.validation_required': 'Pending validation',
    'library.loading': 'Loading motion list…',
    'library.load_error': 'Could not load the motion library. Make sure the server is running.',
    'library.summary_capped': 'Showing the first {shown} of {matched} matches (total {total} tokens, {ready} ready) — use search to narrow the results.',
    'library.summary': '{matched} results (total {total} tokens, {ready} ready).',
    'library.unavailable_note_title': 'The six signs unavailable in Isharah:',
    'library.empty': 'No matching results.',
    'library.play': 'Play',
    'library.playing': 'Playing…',

    'results.recognition.title': 'Sign recognition result',
    'results.recognition.dev_details': 'Diagnostic details (developers)',
    'results.recognition.requested_mode': 'Requested mode',
    'results.recognition.resolved_mode': 'Resolved mode',
    'results.recognition.accepted': 'Accepted?',
    'results.recognition.yes': 'Yes',
    'results.recognition.no': 'No',
    'results.recognition.reason': 'Routing reason',
    'results.recognition.predicted_label': 'Predicted label',
    'results.recognition.confidence': 'Confidence',
    'results.recognition.frame_count': 'Frame count',
    'results.recognition.active_segment': 'Active segment',
    'results.recognition.top_k': 'Top candidates',
    'results.recognition.candidate_scores': 'Candidate path scores',
    'results.recognition.motion_label': 'Recognized sign motion',
    'results.recognition.motion_available': 'Available ({tokens})',
    'results.recognition.motion_unavailable': 'Not currently available ({tokens})',
    'results.recognition.retry_message':
      'The recording was not recognized with enough confidence. Please record again with the upper body and both hands clearly visible, a stable camera, good lighting, and the complete sign in the recording.',
    'results.recognition.recognized': 'Recognized: {label}',
    'results.recognition.recognized_confidence': 'Recognized: {label} ({percent}% confidence)',
    'results.recognition.recognized_confidence_avg': 'Recognized: {label} (avg. decode confidence {percent}%)',
    'results.recognition.none': 'No sign was recognized in this clip.',
    'results.recognition.na': '—',

    'results.page_title': 'Your question',
    'results.page_help': 'Here is what we understood and the answer in both text and sign language.',
    'results.your_question': 'Detected question: {question}',
    'results.ask_again': 'Ask another question',

    'results.answer.title': 'Answer',
    'results.answer.simplified_title': 'Example',
    'results.answer.topic': 'Topic: {topics}',
    'results.answer.unavailable_default': 'No educational answer is available for this input.',
    'results.answer.no_gloss': 'The recognizer produced no usable label for this clip.',
    'results.answer.details_summary': 'Sources and additional details',

    'results.sources.title': 'Sources',
    'results.sources.item': '{domain} — {title} (retrieval score {score})',

    'results.tokens.title': 'Sign token sequence',
    'results.tokens.disclaimer': 'This is a concept sequence for avatar planning, not a fully documented Jordanian Sign Language grammar.',

    'results.motions.title': 'Avatar motion status',
    'results.motions.coverage': 'Motions available: {percent}% of sign tokens',
    'results.motions.missing': 'Currently unavailable motions: {tokens}',

    'devmode.summary': 'Advanced options (developers only)',
    'devmode.auto': 'Automatic (default)',
    'devmode.it': 'Force: Jordanian Technical Signs',
    'devmode.karsl': 'Force: KArSL',
    'devmode.continuous': 'Force: continuous signing',

    'final_cta.title': 'Ready to try SignBridge?',
    'final_cta.description': 'Ask your first question, or record a sign, and watch the avatar present it now.',
    'final_cta.button': 'Go to the interactive experience',

    'footer.tagline': 'An AI-powered Jordanian Sign Language learning assistant.',
    'footer.nav_title': 'Sections',
    'footer.resources_title': 'Data sources',
    'footer.built_with': 'A student project from the University of Jordan focused on inclusive and accessible learning.',
    'footer.thanks_link': 'Thanks',
    'footer.contact_title': 'Contact',
    'footer.contact_note': 'For project communication and collaboration.',
    'footer.credit': 'With special thanks to Ahmad and Majed.',
    'footer.rights_note': 'Designed with care for a more inclusive learning experience.',
    'footer.rights': '© {year} SignBridge',
  },
};

// camera-recorder.js (and its plain-Node unit test) import t() from this
// module for translated error/status strings, so every browser-global
// touch here must be guarded — this file must be safely importable
// outside a browser/DOM environment too.
const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';

const listeners = new Set();

function detectInitialLanguage() {
  // Keep the module's non-browser default unchanged for the recorder unit
  // tests; the browser's first-run interface opens in English to match the
  // supplied UI reference, unless the visitor has already chosen a language.
  if (!isBrowser) return 'ar';
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === 'ar' || stored === 'en') return stored;
  } catch {
    // localStorage unavailable — fall through to default.
  }
  return 'ar';
}

let currentLanguage = detectInitialLanguage();

export function getLanguage() {
  return currentLanguage;
}

export function setLanguage(language) {
  if (language !== 'ar' && language !== 'en') return;
  currentLanguage = language;
  if (isBrowser) {
    try {
      window.localStorage.setItem(STORAGE_KEY, language);
    } catch {
      // Ignore storage failures (private browsing, quota, etc.).
    }
    document.documentElement.lang = language;
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr';
    applyTranslations();
  }
  listeners.forEach((listener) => listener(language));
}

export function onLanguageChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Plain lookup + {placeholder} interpolation. Falls back to the key
 * itself if missing, so a translation gap is visible rather than blank. */
export function t(key, vars) {
  const entry = DICTIONARY[currentLanguage]?.[key] ?? DICTIONARY.ar[key] ?? key;
  if (!vars) return entry;
  return entry.replace(/\{(\w+)\}/g, (_match, name) => (name in vars ? String(vars[name]) : `{${name}}`));
}

/** Applies the current language to every element carrying a data-i18n*
 * attribute. Safe to call repeatedly (e.g. after dynamically-created
 * motion-library rows are re-rendered). */
export function applyTranslations(root) {
  if (!isBrowser) return;
  root = root || document;
  root.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.getAttribute('data-i18n'));
  });
  root.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
  });
  root.querySelectorAll('[data-i18n-aria-label]').forEach((el) => {
    el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria-label')));
  });
  root.querySelectorAll('[data-i18n-alt]').forEach((el) => {
    el.setAttribute('alt', t(el.getAttribute('data-i18n-alt')));
  });
  document.title = t('meta.title');
  const description = document.querySelector('meta[name="description"]');
  if (description) description.setAttribute('content', t('meta.description'));
}

// Apply immediately so document.lang/dir are correct before first paint
// of app-driven content (index.html's own lang/dir attributes are the
// synchronous fallback for the very first paint). No-op outside a browser
// (e.g. when this module is imported by a plain-Node unit test).
if (isBrowser) {
  document.documentElement.lang = currentLanguage;
  document.documentElement.dir = currentLanguage === 'ar' ? 'rtl' : 'ltr';
}

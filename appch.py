import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. تحميل النموذج ---
# !!! تم التغيير هنا لاستخدام النموذج القديم بناءً على طلبك
MODEL_PATH = 'heart_disease_model44.joblib' 
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"خطأ: ملف النموذج '{MODEL_PATH}' غير موجود. تأكد من استخدام النموذج المدرب على البيانات النظيفة.")
    st.stop()

# --- 2. إعدادات الصفحة (التصميم القديم البسيط) ---
st.set_page_config(page_title="تنبؤ بأراض القلب", page_icon="❤️", layout="centered")

# عنوان التطبيق
st.title("🩺 نموذج التنبؤ بأمراض القلب")
st.title(" Created By Haider Abduljalil ")

st.write("أدخل بيانات المريض للتنبؤ باحتمالية الإصابة بأمراض القلب.")

# --- 3. المدخلات في الواجهة الرئيسية (التصميم القديم) ---
# تقسيم الواجهة إلى أعمدة لتنظيم أفضل
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("العمر (Age)", 20, 100, 50)
    sex = st.selectbox("الجنس (Sex)", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    cp = st.selectbox("Chest Pain Type - نوع ألم الصدر(CP)", options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])

with col2:
    trestbps = st.slider("Resting Blood Pressure - ضغط الدم الانقباضي وقت الراحة (trestbps)", 90, 200, 120)
    chol = st.slider("  Serum Cholesterol Level - مستوى الكوليسترول في الدم (chol) ", 100, 600, 200)
    fbs = st.selectbox("سكر الدم اثناء الصيام > 120 mg/dl (fbs)", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

with col3:
    restecg = st.selectbox("Resting Electrocardiogram Results - نتائج تخطيط القلب الكهربائي وقت الراحة (restecg)", options=[("Normal", 0), ("ST-T wave abnormality", 1), ("Hypertrophy", 2)], format_func=lambda x: x[0])
    thalach = st.slider("أقصى معدل نبض (أثناء اختبار الجهد) - Maximum Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("ذبحة صدرية مع التمرين (exang)", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

# مدخلات إضافية في صف جديد
oldpeak = st.slider("قيمة انخفاض مقطع ST(oldpeak)", 0.0, 6.2, 1.0)
slope = st.selectbox("ميل المقطع ST في ذروة التمرين (Slope of the Peak Exercise ST Segment)", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])
    
# --- ⬇️ استخدام الخيارات المتوافقة مع النموذج القديم ⬇️ ---
ca = st.selectbox("عدد الشرايين المسدودة(ca)", [0, 1, 2, 3])
thal = st.selectbox("Thallium Stress Test Result - نتيجة اختبار إجهاد الثاليوم النووي", options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)], format_func=lambda x: x[0])
# --- ⬆️ استخدام الخيارات المتوافقة مع النموذج القديم ⬆️ ---


# --- 4. زر التنبؤ والنتيجة ---
if st.button("الحصول على التنبؤ", type="primary"):
    # --- تحويل القيم إلى أرقام (النسخة القديمة) ---
    sex_val = sex[1]
    cp_val = cp[1]
    fbs_val = fbs[1]
    restecg_val = restecg[1]
    exang_val = exang[1]
    slope_val = slope[1]
    
    # إرسال القيم الخاطئة (1, 2, 3) للنموذج
    thal_val = thal[1]
    
    # تجميع البيانات في مصفوفة بنفس الترتيب الذي تدرب عليه النموذج
    input_data = np.array([[
        age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
        thalach, exang_val, oldpeak, slope_val, ca, thal_val
    ]])

    # إجراء التنبؤ
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # عرض النتيجة (التصميم القديم البسيط)
    st.subheader("نتائج التنبؤ:")
    if prediction[0] == 1:
        st.error(f"**النتيجة: يوجد احتمالية عالية للإصابة بمرض في القلب.** (احتمال بنسبة {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"**النتيجة: الاحتمالية منخفضة للإصابة بمرض في القلب.** (احتمال بنسبة {probability[0][0]*100:.2f}%)")

# --- 5. قسم إخلاء المسؤولية ---
st.markdown("---")
st.warning("""
     تنبيه: هذا النموذج هو أداة تعليمية وتجريبية ولا يغني عن الاستشارة الطبية المتخصصة. 
    النتائج المقدمة هي تنبؤات بناءً على البيانات المدخلة ولا يجب اعتبارها تشخيصًا نهائيًا.
""")

st.markdown("<p style='text-align: center; font-size: 24px ; color: #000000;'>Created by Haider Abdul Jalil</p>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📬  اكتب ملاحظة او رسالة")

# نستخدم st.form لجمع المدخلات قبل إرسالها
with st.form(key='contact_form'):
    
    # --- (تم إلغاء حقل إيميل المستخدم من هنا) ---
    
    # حقل الرسالة
    message_text = st.text_area("اكتب رسالتك هنا...", height=150)
    
    # زر الإرسال
    submit_button = st.form_submit_button(label='إرسال الرسالة')

# هذا الكود يتم تنفيذه فقط عند الضغط على زر الإرسال
if submit_button:
    if not message_text:
        st.warning("الرجاء كتابة رسالة قبل الإرسال.")
    else:
        # استيراد المكتبات اللازمة
        import smtplib
        import ssl
        from email.message import EmailMessage

        # --- قراءة الأسرار المخزنة ---
        # نقرأ الإيميل وكلمة المرور من "الأسرار" التي حفظناها
        try:
            SENDER_EMAIL = st.secrets["email"]
            SENDER_PASSWORD = st.secrets["password"]
            # هذا هو الإيميل الذي ستصل إليه الرسائل (إيميلك أيضًا)
            RECEIVER_EMAIL = st.secrets["email"]
        except KeyError:
            st.error("خطأ في إعدادات الخادم: لم يتم العثور على أسرار البريد الإلكتروني.")
            st.stop()

        # --- تجهيز الرسالة ---
        msg = EmailMessage()
        msg['Subject'] = f"رسالة جديدة من تطبيق أمراض القلب"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        
        # --- (تم تعديل محتوى الرسالة هنا) ---
        body = f"""
        لقد تلقيت رسالة جديدة من تطبيق Streamlit:
        
        الرسالة:
        {message_text}
        """
        msg.set_content(body)

        # --- إرسال الإيميل ---
        try:
            # إنشاء اتصال آمن بـ Gmail
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)
            
            st.success("تم إرسال رسالتك بنجاح! شكرًا لك.")
        
        except Exception as e:
            st.error(f"عفوًا، حدث خطأ أثناء محاولة إرسال الرسالة: {e}")









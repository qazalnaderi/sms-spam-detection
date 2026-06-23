import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SpatialDropout1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------
# تنظیمات اولیه پروژه
# ---------------------

# این عدد را ثابت می گذا# پوشه‌ای برای ذخیره خروجی‌های تصویری گزارش می‌سازیم
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True) 
# تا هر بار اجرای کد تقریبا نتیجه مشابه بدهد
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

path = "./spam.csv"

# فقط پرتکرارترین 10000 کلمه را برای مدل نگه می داریم
# چون کلمه های خیلی کم تکرار معمولا اطلاعات زیادی به مدل نمی دهند
# و فقط اندازه واژگان را بزرگ تر می کنند
maxWords = 10000

# همه پیامک ها را به طول ثابت می رسانیم
# چون مدل ورودی با اندازه ثابت می خواهد
mLength = 40

# مدل ما چند لایه اصلی دارد:
# اول Embedding 
# که کلمه ها را از عدد ساده به بردار عددی تبدیل می کند
# بعد LSTM که ترتیب کلمه ها را در پیام بررسی می کند
# بعد چند لایه Dense و Dropout داریم که برای تصمیم گیری نهایی و کم کردن overfitting استفاده می شوند.
# sigmoidخروجی آخر هم با 
# یک عدد بین 0 و 1 می دهد که احتمال اسپم بودن پیام است

# اندازه بردار عددی هر کلمه در لایه Embedding ا
dim=64

# تعداد نمونه هایی که در هر batch وارد مدل می شوند
batchSz = 32

# epochدر هر 
# مدل یک بار همه داده های آموزشی را می بیند
epochs = 8


# ---------------------
# پاکسازی متن پیامک ها
# ---------------------

def clean_text(message):
    # اگر متن خالی باشد، آن را رشته خالی برمی گردانیم
    # این کار باعث می شود بعدا موقع پردازش متن خطا نگیریم
    if pd.isna(message):
        return ""

    # متن را تبدیل به استرینگ می کنیم و همه حروف را کوچک می کنیم
    message = str(message).lower()

    
    # لینک ها معمولا در پیامک های اسپم زیاد دیده می شوند
    # ولی خود آدرس لینک ها معمولا طولانی و متفاوت است و مدل را شلوغ می کند
    # برای همین آدرس دقیق را نگه نمی داریم
    # linkو فقط با کلمه  
    # وجود لینک را مشخص می کنیم
    message = re.sub("http[^ ]+|www[^ ]+", " link ", message)

    # کاراکترهای غیرضروری را حذف می کنیم
    # فقط حروف انگلیسی، عددها و فاصله را نگه می داریم
    message = re.sub("[^a-zA-Z0-9 ]", " ", message)

    # چند فاصله پشت سر هم را به یک فاصله تبدیل می‌کنیم
    message = " ".join(message.split())

    return message


# ---------------------
# خواندن و آماده کردن دیتاست
# ---------------------

def read_data():
    # فایل دیتاست را می خوانیم
    dataset = pd.read_csv(path, encoding="latin-1")

    # دیتاست دو ستون دارد: v1, v2
    # v1 برچسب پیام است، یعنی ham یا spam
    # v2 متن خود پیامک است
    # ستون های بعدی در این فایل خالی یا اضافی هستند و لازمشان نداریم
    if "v1" in dataset.columns and "v2" in dataset.columns:
        dataset = dataset[["v1", "v2"]]
        dataset.columns = ["label", "message"]

    # برچسب ها را یکدست می کنیم
    # مثلا اگر فاصله اضافه یا حروف بزرگ وجود داشته باشد، مشکل ایجاد نکند
    dataset["label"] = dataset["label"].str.lower().str.strip()

    # فقط دو برچسب اصلی را نگه می داریم
    # ham پیام عادی، spam پیام تبلیغاتی یا مزاحم
    dataset = dataset[dataset["label"].isin(["ham", "spam"])]

    # برچسب متنی را عددی می کنیم تا مدل بتواند با آن آموزش ببیند
    # را 1 spam 
    # ham را 0 در نظر می گیریم
    dataset["target"] = dataset["label"].map({"ham": 0, "spam": 1})

    # متن پیام ها را پاکسازی می کنیم
    dataset["cMessage"] = dataset["message"].apply(clean_text)

    # اگر پیامی بعد از پاکسازی خالی شد، دیگر برای آموزش مفید نیست
    # برای همین آن را حذف می کنیم و شماره ردیف ها را دوباره مرتب می کنیم
    dataset = dataset[dataset["cMessage"].str.len() > 0].reset_index(drop=True)

    return dataset

dataset = read_data()

# ---------------------
# جدا کردن متن ها و برچسب ها
# ---------------------

smsText = dataset["cMessage"].values
smsLabel = dataset["target"].values

# داده ها را اول به دو بخش تقسیم می کنیم:
# 70 درصد برای آموزش مدل
# و 30 درصد برای ادامه کار
# این 30 درصد را بعدا به 
# validation و test تقسیم می کنیم
# stratify باعث می شود نسبت پیام های spam و ham 
# در هر بخش تقریبا مثل دیتاست اصلی بماند
trainText, tempText, trainLabel, tempLabel = train_test_split(
    smsText,
    smsLabel,
    test_size=0.30,
    random_state=seed,
    stratify=smsLabel
)
# حالا همان 30 درصدی که کنار گذاشته بودیم را نصف می کنیم
# نصف آن برای validation 
# استفاده می شود تا موقع آموزش وضعیت مدل را بررسی کنیم
#  نصف دیگر برای می‌ماندtest 
# تا در پایان مدل را روی داده های جدید ارزیابی کنیم
# stratify اینجا هم کمک می کند 
# نسبت spam و ham در validation و test تقریبا برابر بماند
valText, testText, valLabel, testLabel = train_test_split(
    tempText,
    tempLabel,
    test_size=0.50,
    random_state=seed,
    stratify=tempLabel
)
print("Train:", len(trainText), "Validation:", len(valText), "Test:", len(testText))

# ---------------------
# تبدیل متن به عدد
# ---------------------

# Tokenizer کلمه ها را به شماره تبدیل می کند
# oov_token برای کلمه هایی است که در آموزش دیده نشده اند
sTokenizer = Tokenizer(num_words=maxWords, oov_token="<OOV>")

# فقط روی داده آموزشی fit می کنیم
sTokenizer.fit_on_texts(trainText)

# حالا متن‌های هر بخش را به دنباله عددی تبدیل می‌کنیم
# یعنی هر کلمه با شماره‌ای که Tokenizer برای آن ساخته جایگزین می‌شود
trainSeq = sTokenizer.texts_to_sequences(trainText)
valSeq = sTokenizer.texts_to_sequences(valText)
testSeq = sTokenizer.texts_to_sequences(testText)

# پیام ها طول یکسان ندارند
# ولی مدل باید ورودی هایی با طول ثابت بگیرد
# padding پیام های کوتاه را با صفر پر می کند
# و truncating پیام های خیلی بلند را کوتاه می کند
# در نتیجه همه پیام ها به طول mLength می رسند
trainPad = pad_sequences(trainSeq, maxlen=mLength, padding="post", truncating="post")
valPad = pad_sequences(valSeq, maxlen=mLength, padding="post", truncating="post")
testPad = pad_sequences(testSeq, maxlen=mLength, padding="post", truncating="post")

# Tokenizer تعداد کلمه هایی که واقعا داخل 
# ساخته شده را حساب می کنیم
# اگر تعدادشان از maxWords کمتر باشد، همان تعداد واقعی را استفاده می کنیم
# +1 هم به خاطر این است که 
# اندیس صفر معمولا برای padding نگه داشته می شود
vCount = min(maxWords, len(sTokenizer.word_index) + 1)
print("Vocabulary size:", vCount)

# ---------------------
# وزن دادن به کلاس ها
# ---------------------

# ham تعداد پیام های 
# معمولا خیلی بیشتر از spam است
# اگر این نامتعادلی را در نظر نگیریم
# مدل ممکن است بیشتر به سمت ham متمایل شود
# کمک می کندclass_weight
# مدل به پیام های spam هم توجه بیشتری داشته باشد
classVals = np.unique(trainLabel)

weightVals = compute_class_weight(
    class_weight="balanced",
    classes=classVals,
    y=trainLabel
)

spamWeight = {
    int(cls): float(w)
    for cls, w in zip(classVals, weightVals)
}

# ---------------------
# ساخت مدل
# ---------------------

def make_lstm_model():
    # مدل را به صورت Sequential می سازیم
    # چون لایه ها پشت سر هم قرار می گیرند
    model = Sequential()

    # اولین لایه مدل، Embedding است
    # اینجا مشخص می کنیم چند کلمه وارد مدل می شود
    # و برای هر کلمه یک بردار چندتایی ساخته شود
    model.add(Embedding(
        input_dim=vCount,       # تعداد کلمه های واژگان
        output_dim=dim,         # اندازه بردار هر کلمه
        input_length=mLength,   # طول ثابت هر پیامک
        mask_zero=True          # صفرهای padding در یادگیری نادیده گرفته شوند
    ))

    # Embedding بعد از  
    # از SpatialDropout1D استفاده می کنیم
    # در زمان آموزش، بخشی از بردارهای کلمه به طور تصادفی نادیده گرفته می شوند
    # این کار کمک می کند مدل متن های آموزشی را بیش از حد حفظ نکند
    model.add(SpatialDropout1D(0.2))

    # LSTM بخش اصلی مدل برای فهمیدن ترتیب کلمه هاست
    # چون در پیامک فقط وجود کلمه مهم نیست
    # ترتیب و کنار هم آمدن کلمه ها هم اثر دارد
    # dropout و recurrent_dropout هم 
    # برای کم کردن overfitting داخل همین لایه استفاده شده اند
    model.add(LSTM(64, dropout=0.25, recurrent_dropout=0.25))

    # خروجی LSTM را وارد یک لایه Dense می کنیم
    # Dense یعنی هر نورون این لایه به خروجی های لایه قبلی وصل است
    # این لایه ویژگی های یاد گرفته شده را ترکیب می کند تا تصمیم گیری نهایی بهتر انجام شود
    model.add(Dense(32, activation="relu"))

    # دوباره Dropout می گذاریم تا مدل به چند ویژگی خاص بیش از حد وابسته نشود
    model.add(Dropout(0.4))

    # لایه آخر فقط یک خروجی می دهد
    # sigmoid خروجی را بین 0 و 1 نگه می دارد
    # عدد نزدیک به 1 یعنی احتمال spam بودن پیام بیشتر است
    model.add(Dense(1, activation="sigmoid"))

    # مدل را برای مسئله دوکلاسه آماده می کنیم
    # چون خروجی فقط ham یا spam است
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = make_lstm_model()

callList = [
    # EarlyStopping روند آموزش را کنترل می کند
    # validation اگر خطای  
    # برای چند epoch بهتر نشود
    # آموزش را متوقف می کند
    # این کار کمک می کند مدل بی دلیل روی داده آموزشی ادامه ندهد
    # و احتمال overfitting کمتر شود
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        restore_best_weights=True
    )
]

# ---------------------
# آموزش مدل
# ---------------------

# مدل با داده‌های آموزشی train می‌شود
# validation_data برای بررسی عملکرد مدل در حین آموزش است
# callbacks شامل EarlyStopping است تا آموزش بی‌دلیل ادامه پیدا نکند
# class_weight هم برای توجه بیشتر به کلاس کم‌تعدادتر استفاده می‌شود
hist = model.fit(
    trainPad,
    trainLabel,
    validation_data=(valPad, valLabel),
    epochs=epochs,
    batch_size=batchSz,
    callbacks=callList,
    class_weight=spamWeight,
    verbose=1
)

# ---------------------
# test ارزیابی مدل روی داده 
# ---------------------

# مدل را روی داده های test ارزیابی می کنیم
# این داده ها در آموزش استفاده نشده اند
# پس نتیجه نشان می دهد مدل روی پیام های جدید چقدر خوب عمل می کند
# verbose=0 یعنی خروجی اضافه موقع ارزیابی چاپ نشود
testLoss, testAcc = model.evaluate(testPad, testLabel, verbose=0)

# خروجی مدل احتمال spam بودن است
predProb = model.predict(testPad, verbose=0).flatten()

# اگر احتمال از 0.5 بیشتر باشد
# پیام را spam می گیریم
predLabel = (predProb >= 0.5).astype(int)

print("Test Accuracy:", round(testAcc, 4))
print("\nTest Loss:", testLoss)
print("Accuracy Score:", accuracy_score(testLabel, predLabel))
print("\nClassification Report:")
print(classification_report(testLabel, predLabel, target_names=["Ham", "Spam"]))

# ---------------------
# ذخیره خروجی‌های تصویری گزارش
# ---------------------

# ماتریس درهم‌ریختگی
cm = confusion_matrix(testLabel, predLabel)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


# نمودار دقت آموزش و اعتبارسنجی
plt.figure(figsize=(8, 5))
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "training_validation_accuracy.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


# نمودار خطای آموزش و اعتبارسنجی
plt.figure(figsize=(8, 5))
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "training_validation_loss.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# ---------------------
# تست دستی یک پیامک
# ---------------------

def check_sms(message, threshold=0.5):
    # اول متن پیام را مثل داده های آموزشی پاکسازی می کنیم
    cmessage = clean_text(message)

    # متن را به دنباله عددی تبدیل می کنیم
    mSeq = sTokenizer.texts_to_sequences([cmessage])

    # پیام را به طول ثابت mLength می رسانیم
    # اگر پیام کوتاه باشد، با padding="post" صفرها به آخر پیام اضافه می شوند
    # اگر پیام بلندتر باشد، با truncating="post" قسمت های اضافه از آخر پیام حذف می شوند
    mPad = pad_sequences(mSeq, maxlen=mLength, padding="post", truncating="post")

    # مدل احتمال spam بودن پیام را برمی گرداند
    pSpam = float(model.predict(mPad, verbose=0)[0][0])
    # چون فقط دو کلاس داریم
    # احتمال ham برابر است با 1 منهای احتمال spam
    pHam = 1 - pSpam

    # threshold همان حد تصمیم گیری مدل است
    # اگر احتمال spam بودن پیام از این مقدار بیشتر یا برابر باشد
    # پیام را spam در نظر می گیریم
    # در غیر این صورت پیام ham حساب می شود
    if pSpam >= threshold:
        flLabel = "Spam"
        score = pSpam
    else:
        flLabel = "Ham"
        score = pHam

    # برای خواناتر شدن خروجی، اعتماد مدل را به سه سطح تقسیم می کنیم
    if score >= 0.80:
        confText = "High"
    elif score >= 0.65:
        confText = "Medium"
    else:
        confText = "Low"

    print("Message:", message)
    print("Prediction:", flLabel)
    print("Spam probability:", round(pSpam, 4))
    print("Ham probability:", round(pHam, 4))
    print("Confidence:", confText)

    return flLabel, pSpam


# ---------------------
# چند پیام تستی برای بررسی دستی مدل
# ---------------------

testSmsList = [
    "Congratulations! You have won a free iPhone. Click this link now to claim your prize.",
    "Hey, are we still meeting at 6 today?",
    "URGENT! Your account has been selected for a cash reward. Reply WIN to receive it.",
    "Can you send me the homework file when you get home?",
    "You have been chosen for a free vacation. Call this number now to confirm your booking.",
    "Please call me when you finish your class.",
    "Your bank account needs verification. Please login to update your information.",
    "Don't forget to bring your notebook tomorrow.",
]

print("\nManual SMS tests")
print("=" * 60)

for sms in testSmsList:
    print("\n")
    check_sms(sms)
    print("-" * 60)


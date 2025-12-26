import os
import json
import random
import re

from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

# ======================
# ENV
# ======================
TOKEN = os.environ["TOKEN"]


# ======================
# FILES
# ======================
STYLE_MEMORY_FILE = "style_memory.json"
WORD_WEIGHTS = defaultdict(Counter)


# ======================
# STATE
# ======================
LEARNING_MODE = False
STYLE_MEMORY = []
MAX_STYLE_MEMORY = 1000


# ======================
# TRAINING DATA (❌ دست نزن)
# ======================

TRAINING_DATA = [
    ("سلام", ["سلام ", "سلام چطوری ", "السلام علی یا حضرت روشن"]),
    ("جکسن", ["ها ", "کیر خر ", "خب کیر","جان","بله"]),
    ("کمک", ["چه رخ داد", "ریدم چی شد","بگا رفتیم"]),
    ("جکسن خوبی؟", ["اره تو چی", "مرسی تو خوبی", "خوبیو کیر خر خب نه"]),
]

# ======================
# LOAD / SAVE STYLE MEMORY
# ======================
def load_style_memory():
    global STYLE_MEMORY
    if os.path.exists(STYLE_MEMORY_FILE):
        try:
            with open(STYLE_MEMORY_FILE, "r", encoding="utf-8") as f:
                STYLE_MEMORY = json.load(f)
            print(f"Loaded {len(STYLE_MEMORY)} style messages")
        except Exception as e:
            print("Failed to load style memory:", e)


def save_style_memory():
    try:
        with open(STYLE_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(STYLE_MEMORY, f, ensure_ascii=False)
    except Exception as e:
        print("Failed to save style memory:", e)


load_style_memory()


# ======================
# MODEL
# ======================
def rebuild_model():
    global vectorizer, X, questions, answers

    questions = []
    answers = []

    for q, ans_list in TRAINING_DATA:
        questions.append(q)
        answers.append(ans_list)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)


rebuild_model()


def find_best_answer(text, threshold=0.35):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, X)[0]

    idx = sims.argmax()
    if sims[idx] < threshold:
        return None

    return random.choice(answers[idx])

def extract_subject(text):
    words = text.split()
    if not words:
        return None
    return words[0]


# ======================
# STYLE LEARNING
# ======================
def valid_style_message(text):
    text = text.strip()
    if len(text.split()) < 2 or len(text.split()) > 20:
        return False
    blacklist = ["http", "@", "/", "join"]
    return not any(b in text.lower() for b in blacklist)


def generate_style_reply():
    if not STYLE_MEMORY:
        return None
    return random.choice(STYLE_MEMORY)

def generate_weighted_opinion(subject):
    if subject not in WORD_WEIGHTS:
        return None

    common = WORD_WEIGHTS[subject].most_common(3)
    if not common:
        return None

    top_word = common[0][0]

    templates = [
        f"به نظر جمع، {subject} بیشتر {top_word} حساب میشه",
        f"اکثراً میگن {subject} {top_word} ـه",
        f"نظر غالب اینه که {subject} {top_word} ـه"
    ]

    return random.choice(templates)

# ======================
# ADDRESSING
# ======================
BOT_NAMES = ["جکسن"]


def is_addressed(update, context):
    text = update.message.text.lower().strip()

    if update.message.reply_to_message:
        return update.message.reply_to_message.from_user.id == context.bot.id

    if context.bot.username and context.bot.username.lower() in text:
        return True

    for name in BOT_NAMES:
        pattern = rf"^(?:[\W_]*)(?:{re.escape(name)})(?:\b|[،,: ])"
        if re.search(pattern, text):
            return True

    return False


# ======================
# COMMANDS
# ======================
async def learn_on(update, context):
    global LEARNING_MODE
    LEARNING_MODE = True
    await update.message.reply_text("روشن شدم ولی کصباز نه")


async def learn_off(update, context):
    global LEARNING_MODE
    LEARNING_MODE = False
    save_style_memory()
    await update.message.reply_text("آقا تمام")


async def learn_status(update, context):
    await update.message.reply_text(
        f"📊 وضعیت یادگیری:\n"
        f"فعال: {LEARNING_MODE}\n"
        f"تعداد پیام‌های لحن: {len(STYLE_MEMORY)}"
    )
    # ======================
# MESSAGE HANDLER
# ======================
async def message_handler(update, context):
    global STYLE_MEMORY

    text = update.message.text

    # 1) PASSIVE LEARNING (همه‌ی پیام‌ها)
    if LEARNING_MODE and not update.message.from_user.is_bot:
        if valid_style_message(text):
            STYLE_MEMORY.append(text)

            subject = extract_subject(text)
            if subject:
                for w in text.split():
                    if len(w) > 2:
                    WORD_WEIGHTS[subject][w] += 1

            if len(STYLE_MEMORY) > MAX_STYLE_MEMORY:
                STYLE_MEMORY.pop(0)

            # ذخیره‌ی دوره‌ای
            if len(STYLE_MEMORY) % 20 == 0:
                save_style_memory()

    # 2) فقط وقتی صدا شده جواب بده
    if not is_addressed(update, context):
        return
    
    subject = extract_subject(text)
    weighted = generate_weighted_opinion(subject)
    if weighted:
        await update.message.reply_text(weighted)
        return
    
    # 3) جواب
    answer = find_best_answer(text)
    if answer:
        await update.message.reply_text(answer)
        return

    style_reply = generate_style_reply()
    if style_reply:
        await update.message.reply_text(style_reply)


# ======================
# MAIN
# ======================
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("learn_on", learn_on))
    app.add_handler(CommandHandler("learn_off", learn_off))
    app.add_handler(CommandHandler("learn_status", learn_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

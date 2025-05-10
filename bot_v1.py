import os
from uuid import uuid4
import cv2
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)

TOKEN = "7594137884:AAHvigig7KcStYeVcubHrZ0Na4ftrfUyZT0"
input_dir = "/Users/stanislavmalarcuk/Desktop/videos/input"
output_dir = "/Users/stanislavmalarcuk/Desktop/videos/output"


async def main_menu(update_or_query, context):
    keyboard = [
        [InlineKeyboardButton("🚀 Старт", callback_data="start_action")],
        [InlineKeyboardButton("ℹ️ Инфо", callback_data="info_menu")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if isinstance(update_or_query, Update) and update_or_query.message:
        await update_or_query.message.reply_text("Привет! Выбери действие:", reply_markup=reply_markup)
    else:
        await update_or_query.edit_message_text("Привет! Выбери действие:", reply_markup=reply_markup)


# Подменю "Инфо"
async def info_menu(query):
    keyboard = [
        [InlineKeyboardButton("🤖 О боте", callback_data="about_bot")],
        [InlineKeyboardButton("👨‍💻 О разработчиках", callback_data="about_devs")],
        [InlineKeyboardButton("📘 О проекте", callback_data="about_project")],
        [InlineKeyboardButton("🔙 Назад", callback_data="go_back")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("ℹ️ Выбери интересующий пункт:", reply_markup=reply_markup)


# Команда /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await main_menu(update, context)


# Обработка нажатий кнопок
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    match query.data:
        case "start_action":
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="go_back")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text("📤 Пожалуйста, пришлите видео до 2 минут.", reply_markup=reply_markup)

        case "info_menu":
            await info_menu(query)

        case "about_bot":
            await query.edit_message_text(
                text="🤖 Я бот, который принимает видео и рассказывает о проекте.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="info_menu")]])
            )

        case "about_devs":
            await query.edit_message_text(
                text="👨‍💻 Бот разработан талантливыми Python-разработчиками.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="info_menu")]])
            )

        case "about_project":
            await query.edit_message_text(
                text="📘 Этот проект создан для демонстрации Telegram-ботов.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="info_menu")]])
            )

        case "go_back":
            await main_menu(query, context)

        case _:
            await query.edit_message_text("❓ Неизвестная команда.")


# Обработка полученного видео
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video = update.message.video

    if video.duration > 120:
        await update.message.reply_text("⚠️ Видео слишком длинное. Пришли видео до 2 минут.")
        return

    os.makedirs(input_dir, exist_ok=True)  # создаём папку, если не существует
    os.makedirs(output_dir, exist_ok=True)  # создаём папку для output, если не существует

    # Уникальное имя файла
    file_name = f"{uuid4()}.mp4"
    full_path = os.path.join(input_dir, file_name)

    # Скачиваем видео на диск
    file = await context.bot.get_file(video.file_id)
    await file.download_to_drive(full_path)

    await update.message.reply_text("✅ Видео сохранено!")

    # Обрабатываем видео: преобразуем в черно-белое
    output_path = os.path.join(output_dir, "1.mp4")



    while (1):
        if (len(os.listdir(output_dir)) == 1):
            continue
        else:

            await send_video_to_user(update.effective_chat.id, output_path, context)
            break

    os.remove(output_path)
    os.remove(full_path)
async def send_video_to_user(chat_id: int, video_path: str, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет видео пользователю по chat_id и удаляет видео после отправки."""
    if not os.path.isfile(video_path):
        print(f"Файл {video_path} не найден.")
        return

    try:
        with open(video_path, 'rb') as video_file:
            await context.bot.send_video(
                chat_id=chat_id,
                video=InputFile(video_file),
                caption="🎬 Вот ваше черно-белое видео:"
            )



    except Exception as e:
        print(f"Ошибка при отправке видео: {e}")






# Запуск
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    app.run_polling()

import os
from uuid import uuid4
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)

TOKEN = "7594137884:AAHvigig7KcStYeVcubHrZ0Na4ftrfUyZT0"
input_dir = "/Users/stanislavmalarcuk/Desktop/videos/input"
output_dir = "/Users/stanislavmalarcuk/Desktop/videos/output"


class Bot:
    def __init__(self, token: str):
        self.token = token
        self.app = ApplicationBuilder().token(self.token).build()

        # Добавляем обработчики
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
        self.app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, self.media_handler))  # Обработка фото и видео

    async def main_menu(self, update_or_query, context):
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
    async def info_menu(self, query):
        keyboard = [
            [InlineKeyboardButton("🤖 О боте", callback_data="about_bot")],
            [InlineKeyboardButton("👨‍💻 О разработчиках", callback_data="about_devs")],
            [InlineKeyboardButton("📘 О проекте", callback_data="about_project")],
            [InlineKeyboardButton("🔙 Назад", callback_data="go_back")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("ℹ️ Выбери интересующий пункт:", reply_markup=reply_markup)

    # Команда /start
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.main_menu(update, context)

    # Обработка нажатий кнопок
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        match query.data:
            case "start_action":
                keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="go_back")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text("📤 Пожалуйста, пришлите видео до 2 минут.", reply_markup=reply_markup)

            case "info_menu":
                await self.info_menu(query)

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
                await self.main_menu(query, context)

            case _:
                await query.edit_message_text("❓ Неизвестная команда.")

    # Универсальная обработка медиа (фото и видео)
    async def media_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Определяем тип медиа (фото или видео)
        if update.message.video:
            media = update.message.video
            file_ext = "mp4"
        elif update.message.photo:
            media = update.message.photo[-1]  # Берем самое большое изображение
            file_ext = "jpg"
        else:
            await update.message.reply_text("❓ Не поддерживаемый формат.")
            return

        os.makedirs(input_dir, exist_ok=True)  # создаём папку, если не существует
        os.makedirs(output_dir, exist_ok=True)  # создаём папку для output, если не существует

        # Уникальное имя файла
        file_name = f"{uuid4()}.{file_ext}"
        full_path = os.path.join(input_dir, file_name)

        # Скачиваем медиа на диск
        file = await context.bot.get_file(media.file_id)
        await file.download_to_drive(full_path)

        await update.message.reply_text(f"✅ {file_ext.upper()} сохранено!")

        # Отправляем медиа обратно пользователю
        await self.send_media_to_user(update.effective_chat.id, full_path, file_ext, context)

        os.remove(full_path)

    # Универсальная функция отправки медиа (фото и видео)
    async def send_media_to_user(self, chat_id: int, media_path: str, file_ext: str, context: ContextTypes.DEFAULT_TYPE):
        """Отправляет медиа (фото или видео) пользователю по chat_id и удаляет его после отправки."""
        if not os.path.isfile(media_path):
            print(f"Файл {media_path} не найден.")
            return

        try:
            with open(media_path, 'rb') as media_file:
                if file_ext == "mp4":
                    await context.bot.send_video(
                        chat_id=chat_id,
                        video=InputFile(media_file),
                        caption="🎬 Вот ваше видео:"
                    )
                elif file_ext == "jpg":
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=InputFile(media_file),
                        caption="🖼 Вот ваше изображение:"
                    )
        except Exception as e:
            print(f"Ошибка при отправке медиа: {e}")

    def run(self):
        self.app.run_polling()


if __name__ == "__main__":
    bot = Bot(TOKEN)
    bot.run()

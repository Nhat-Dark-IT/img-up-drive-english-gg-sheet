# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API Settings
GOOGLE_DRIVE_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
GOOGLE_SHEETS_ID = os.getenv('GOOGLE_SHEETS_ID')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# OCR Settings 
TESSERACT_PATH = os.getenv('TESSERACT_PATH')

# Google Drive folder structure
FOLDER_NAMES = {
    'daily': 'Từ vựng daily',
    'synonyms': 'Từ đồng nghĩa',
    'toeic': 'TOEIC',
    'ielts': 'IELTS'
}

# Google Sheets structure
SHEET_NAMES = {
    'daily': 'Từ vựng daily',
    'synonyms': 'Từ đồng nghĩa', 
    'toeic': 'TOEIC',
    'ielts': 'IELTS'
}

# Sheet columns
SHEET_COLUMNS = [
    'STT',
    'Tên ảnh xử lý',
    'Tiêu đề',
    'Từ Vựng',
    'IPA',
    'Nghĩa Tiếng Việt',
    'Nghĩa Tiếng Anh',
    'Từ đồng nghĩa'
]

# Image processing settings
IMAGE_PROCESS = {
    'min_confidence': 60,  # Minimum OCR confidence score
    'resize_width': 1280,  # Width to resize images to
    'blur_kernel': (3,3),  # Gaussian blur kernel size
    'contrast_limit': 3,   # Contrast enhancement limit
}

# Automation schedule (24h format)
SCHEDULE_TIMES = [
    '07:00',  # Morning
    '12:00',  # Noon
    '20:00'   # Evening
]
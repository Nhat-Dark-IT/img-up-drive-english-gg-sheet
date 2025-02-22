# src/sheets_handler.py
import os
import sys
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from googleapiclient.errors import HttpError
# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.settings import SHEET_NAMES, SHEET_COLUMNS, GOOGLE_SHEETS_ID
from src.auth_helper import get_google_service

class SheetsHandler:
    def __init__(self):
        """Khởi tạo Google Sheets service."""
        """Khởi tạo với rate limiting."""
        self.service = get_google_service('sheets', 'v4')
        self.spreadsheet_id = GOOGLE_SHEETS_ID
        self.requests_per_minute = 50  # Giới hạn dưới 60
        self.last_request_time = time.time()
        self.request_count = 0

    def get_sheet_id(self, sheet_type):
        """Lấy sheet ID theo loại."""
        sheet_name = SHEET_NAMES.get(sheet_type)
        if not sheet_name:
            raise ValueError(f"Invalid sheet type: {sheet_type}")

        # Lấy thông tin sheets
        sheet_metadata = self.service.spreadsheets().get(
            spreadsheetId=self.spreadsheet_id
        ).execute()

        # Tìm sheet ID
        for sheet in sheet_metadata.get('sheets', []):
            if sheet['properties']['title'] == sheet_name:
                return sheet['properties']['sheetId']
        return None
    def _handle_rate_limit(self):
        """Xử lý giới hạn request."""
        current_time = time.time()
        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time

        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                print(f"Đợi {sleep_time:.2f}s do đạt giới hạn request")
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()

        self.request_count += 1

    @retry(
        retry=retry_if_exception_type(HttpError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def batch_append_vocabulary(self, sheet_type, vocabulary_data):
        """Thêm nhiều từ vựng trong một request."""
        try:
            sheet_name = SHEET_NAMES.get(sheet_type)
            if not sheet_name:
                raise ValueError(f"Invalid sheet type: {sheet_type}")

            # Lấy tất cả số thứ tự trong một request
            self._handle_rate_limit()
            range_name = f"{sheet_name}!A:A"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            # Tính toán số thứ tự cho tất cả entries
            values = result.get('values', [])
            next_number = max([
                int(row[0]) for row in values 
                if row and row[0].isdigit()
            ], default=0) + 1

            # Chuẩn bị dữ liệu hàng loạt
            rows = []
            for item in vocabulary_data:
                row = [
                    next_number,
                    item.get('image_name', ''),
                    item.get('title', ''),
                    item.get('word', ''),
                    item.get('ipa', ''),
                    item.get('vn_meaning', ''),
                    item.get('en_meaning', ''),
                    item.get('synonyms', '')
                ]
                rows.append(row)
                next_number += 1

            # Thêm dữ liệu trong một request
            self._handle_rate_limit()
            range_name = f"{sheet_name}!A:H"
            body = {'values': rows}

            response = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()

            print(f"✅ Đã thêm {len(rows)} dòng vào {sheet_name}")
            return response

        except HttpError as e:
            if e.resp.status == 429:  # Rate limit exceeded
                print("Vượt giới hạn request, thử lại sau...")
                raise
            raise
    def append_vocabulary(self, sheet_type, vocabulary_data):
        """Thêm từ vựng vào sheet với xử lý batch và rate limit."""
        try:
            sheet_name = SHEET_NAMES.get(sheet_type)
            if not sheet_name:
                raise ValueError(f"Invalid sheet type: {sheet_type}")
    
            # Rate limit check
            self._handle_rate_limit()
    
            # Get all numbers in one request
            range_name = f"{sheet_name}!A:A"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
    
            # Calculate next number
            values = result.get('values', [])
            next_number = max([
                int(row[0]) for row in values 
                if row and row[0].isdigit()
            ], default=0) + 1
    
            # Prepare batch data
            rows = []
            for item in vocabulary_data:
                row = [
                    next_number,
                    item.get('image_name', ''),
                    item.get('title', ''),
                    item.get('word', ''),
                    item.get('ipa', ''),
                    item.get('vn_meaning', ''),
                    item.get('en_meaning', ''),
                    item.get('synonyms', '')
                ]
                rows.append(row)
                next_number += 1
    
            # Rate limit before append
            self._handle_rate_limit()
    
            # Batch append with retry
            @retry(
                retry=retry_if_exception_type(HttpError),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10)
            )
            def append_batch():
                range_name = f"{sheet_name}!A:H"
                body = {'values': rows}
                
                print(f"📝 Ghi vào {sheet_name}: {len(rows)} dòng")
                
                response = self.service.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                
                print(f"✅ Đã ghi {len(rows)} dòng vào {sheet_name}")
                return response
    
            return append_batch()
    
        except HttpError as e:
            if e.resp.status == 429:
                print("⚠️ Rate limit exceeded, retrying...")
                raise
            print(f"❌ Lỗi khi ghi dữ liệu vào {sheet_name}: {e}")
            raise


    def get_next_number(self, sheet_name):
        """Lấy số thứ tự tiếp theo trong sheet."""
        range_name = f"{sheet_name}!A:A"
        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        if not values:
            return 1
            
        # Lọc các số thứ tự hợp lệ
        numbers = []
        for row in values:
            try:
                num = int(row[0])
                numbers.append(num)
            except (ValueError, IndexError):
                continue
                
        return max(numbers, default=0) + 1

    def format_sheet(self, sheet_id):
        """Format sheet theo mẫu chuẩn."""
        requests = [
            # Đặt headers
            {
                'updateCells': {
                    'rows': [{
                        'values': [{
                            'userEnteredValue': {'stringValue': col}
                        } for col in SHEET_COLUMNS]
                    }],
                    'fields': 'userEnteredValue',
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    }
                }
            },
            # Format header
            {
                'repeatCell': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                            'textFormat': {'bold': True},
                            'horizontalAlignment': 'CENTER'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            }
        ]

        # Execute formatting
        self.service.spreadsheets().batchUpdate(
            spreadsheetId=self.spreadsheet_id,
            body={'requests': requests}
        ).execute()

def test_sheets_handler():
    """Test sheets handler functionality."""
    try:
        handler = SheetsHandler()
        
        # Test với sample data
        sample_data = [{
            'image_name': 'test_image.jpg',
            'title': 'STUDY4',
            'word': 'example',
            'ipa': '/ɪɡˈzæmpəl/',
            'vn_meaning': 'ví dụ',
            'en_meaning': 'a representative form',
            'synonyms': 'instance, sample'
        }]

        # Test các loại sheet
        for sheet_type in SHEET_NAMES.keys():
            print(f"\nTesting {sheet_type} sheet:")
            
            # Test get sheet ID
            sheet_id = handler.get_sheet_id(sheet_type)
            print(f"✓ Found sheet ID: {sheet_id}")
            
            # Test append data
            handler.append_vocabulary(sheet_type, sample_data)
            print(f"✓ Appended test data to {sheet_type} sheet")
            
            # Test format sheet
            handler.format_sheet(sheet_id)
            print(f"✓ Formatted {sheet_type} sheet")

        print("\n✅ All Sheets handler tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Sheets handler test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_sheets_handler()
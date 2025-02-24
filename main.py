# src/main.py
import os
import sys
import time
from datetime import datetime
import schedule
import logging
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.settings import SCHEDULE_TIMES
from src.drive_handler import DriveHandler
from src.image_processor import ImageProcessor
from src.sheets_handler import SheetsHandler

class VocabularyAutomation:
    def __init__(self):
        """Khởi tạo các components."""
        self.drive = DriveHandler()
        self.processor = ImageProcessor()
        self.sheets = SheetsHandler()
        
    def process_folder(self, folder_type):
        """Xử lý một thư mục từ vựng."""
        temp_files = []  # Track all temporary files
        
        try:
            print(f"\nProcessing {folder_type} folder...")
            
            # 1. Lấy ID thư mục gốc
            folder_id = self.drive.get_folder_id(folder_type)
            if not folder_id:
                print(f"Folder not found for type: {folder_type}")
                return
                
            # 2. Tạo/lấy thư mục ngày
            date_folder_id = self.drive.create_date_folder(folder_id)
            
            # 3. Lấy danh sách ảnh
            images = self.drive.list_images(date_folder_id)
            if not images:
                print("No images found to process")
                return
            # Lọc ảnh chưa xử lý
            unprocessed_images = [
                img for img in images 
                if not img['name'].startswith('DONE_')
            ]
            
            if not unprocessed_images:
                print("All images have been processed")
                return
                
            print(f"Found {len(unprocessed_images)} unprocessed images")
            # 4. Xử lý từng ảnh và thu thập data
            vocabulary_data = []
            for image in unprocessed_images:
                try:
                    # Download ảnh
                    temp_path = os.path.join("temp", image['name'])
                    os.makedirs("temp", exist_ok=True)
                    self.drive.download_image(image['id'], temp_path)
                    # Add paths to cleanup list
                    processed_path = temp_path.replace('.jpg', '_processed.jpg')
                    enhanced_path = temp_path.replace('.jpg', '_processed_enhanced.txt')
                    ocr_path = temp_path.replace('.jpg', '_processed_ocr.txt')
                    
                    temp_files.extend([
                        temp_path,
                        processed_path,
                        enhanced_path,
                        ocr_path
                    ])
                    # Xử lý OCR và AI
                    enhanced_text = self.processor.process_image(temp_path, processed_path)
                    
                    # Parse kết quả từ file enhanced
                    entries = self.parse_vocabulary(image['name'], enhanced_text)
                    if entries:
                        vocabulary_data.extend(entries)
                        print(f"Found {len(entries)} entries from {image['name']}")
                    
                    # Đánh dấu đã xử lý
                    processed_name = f"DONE_{image['name']}"
                    self.drive.mark_as_processed(image['id'], processed_name)
                    # Thêm file vào danh sách cần xóa sau khi lưu vào Google Sheets
                    # Cleanup
                    os.remove(temp_path)
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                     
                except Exception as e:
                    print(f"Error processing image {image['name']}: {str(e)}")
                    continue
            
            # 5. Append tất cả data vào sheet
            if vocabulary_data:
                self.sheets.append_vocabulary(folder_type, vocabulary_data)
                print(f"\n✅ Successfully added {len(vocabulary_data)} total entries to {folder_type} sheet")
            else:
                print(f"\nNo entries found to add to {folder_type} sheet")
                
        except Exception as e:
            print(f"Error processing folder {folder_type}: {str(e)}")
            
        finally:
            # Cleanup all temporary files
            print("\nCleaning up temporary files...")
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"🗑️ Deleted: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error deleting file {file_path}: {str(e)}")
            
    
    def parse_vocabulary(self, image_name, enhanced_text):
        """Parse kết quả từ file enhanced text."""
        try:
            # Lấy đường dẫn file enhanced text
            enhanced_path = image_name.replace('.jpg', '_processed_enhanced.txt')
            enhanced_path = os.path.join('temp', enhanced_path)
            
            # Kiểm tra file tồn tại
            if not os.path.exists(enhanced_path):
                print(f"⚠️ Enhanced text file not found: {enhanced_path}")
                return None
                
            # Đọc nội dung file
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Parse từng dòng thành entries
            vocab_entries = []
            for line in lines:
                if not line.strip():
                    continue
                    
                # Split by tab
                parts = line.strip().split('\t')
                
                # Đảm bảo đủ 4 phần tử
                while len(parts) < 4:
                    parts.append('')
                    
                # Create entry với các trường mặc định nếu thiếu
                entry = {
                    'image_name': image_name,
                    'title': parts[0].strip() or 'Undefined',  # Giá trị mặc định nếu trống
                    'word': parts[1].strip() or 'Unknown',     # Giá trị mặc định nếu trống
                    'vn_meaning': parts[2].strip() or 'Chưa có nghĩa',  # Giá trị mặc định nếu trống
                    'en_meaning': parts[3].strip() or 'No meaning'      # Giá trị mặc định nếu trống
                }
                
                # Log warning nếu thiếu thông tin
                missing_fields = []
                if not parts[1].strip(): missing_fields.append('word')
                if not parts[2].strip(): missing_fields.append('vn_meaning') 
                if not parts[3].strip(): missing_fields.append('en_meaning')
                
                if missing_fields:
                    print(f"⚠️ Missing fields {missing_fields} in entry: {line}")
                
                vocab_entries.append(entry)
                    
            return vocab_entries if vocab_entries else None
            
        except Exception as e:
            print(f"❌ Error parsing vocabulary: {str(e)}")
            return None
    
    def run_job(self):
        """Chạy xử lý tất cả loại thư mục."""
        print(f"\nRunning automation job at {datetime.now()}")
        
        folder_types = ['daily', 'synonyms', 'toeic', 'ielts']
        for folder_type in folder_types:
            self.process_folder(folder_type)
            
def main():
    logging.info("Starting vocabulary automation...")
    """Main function để chạy automation."""
    automation = VocabularyAutomation()
    
    # Chạy lần đầu
    automation.run_job()   
    # Keep running

if __name__ == "__main__":
    main()
    
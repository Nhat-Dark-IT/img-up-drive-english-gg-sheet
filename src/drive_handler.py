# src/drive_handler.py
import os
import sys
from datetime import datetime
from googleapiclient.http import MediaIoBaseDownload
import io

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.settings import FOLDER_NAMES, GOOGLE_DRIVE_FOLDER_ID
from src.auth_helper import get_google_service

# Rest of the code remains the same...

class DriveHandler:
    def __init__(self):
        """Initialize drive service and base folder ID."""
        self.service = get_google_service('drive')
        self.base_folder_id = GOOGLE_DRIVE_FOLDER_ID

    def get_folder_id(self, folder_type):
        """Get folder ID by type (daily/synonyms/toeic/ielts)."""
        folder_name = FOLDER_NAMES.get(folder_type)
        if not folder_name:
            raise ValueError(f"Invalid folder type: {folder_type}")

        results = self.service.files().list(
            q=f"'{self.base_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        
        folders = results.get('files', [])
        return folders[0]['id'] if folders else None

    def create_date_folder(self, parent_folder_id):
        """Create folder with today's date if not exists."""
        today = datetime.now().strftime('%d-%m-%Y')
        
        # Check if folder exists
        results = self.service.files().list(
            q=f"'{parent_folder_id}' in parents and name='{today}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        
        if results.get('files'):
            return results['files'][0]['id']
            
        # Create new folder
        file_metadata = {
            'name': today,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        
        folder = self.service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        return folder.get('id')

    def list_images(self, folder_id):
        """List all image files in the specified folder."""
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and mimeType contains 'image/'",
            fields="files(id, name)"
        ).execute()
        
        return results.get('files', [])

    def download_image(self, file_id, output_path):
        """Download an image file by ID."""
        request = self.service.files().get_media(fileId=file_id)
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        with open(output_path, 'wb') as f:
            f.write(fh.read())
            f.close()

    def mark_as_processed(self, file_id, new_name):
        """Rename file to mark as processed."""
        self.service.files().update(
            fileId=file_id,
            body={'name': new_name}
        ).execute()
    def is_processed(self, folder_id, processed_name):
        """Kiểm tra xem file đã được đánh dấu là DONE_ trong Google Drive hay chưa."""
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and name='{processed_name}'",
            fields="files(id, name)"
        ).execute()
        
        return bool(results.get('files', []))


def test_drive_handler():
    """Test drive handler functionality for all folder types."""
    try:
        handler = DriveHandler()
        folder_types = ['daily', 'synonyms', 'toeic', 'ielts']
        
        for folder_type in folder_types:
            print(f"\nTesting {folder_type} folder:")
            
            # Test getting folder ID
            folder_id = handler.get_folder_id(folder_type)
            if not folder_id:
                raise Exception(f"Could not get folder ID for {folder_type}")
            print(f"✓ Found {folder_type} folder: {folder_id}")
            
            # Test creating date folder
            date_folder_id = handler.create_date_folder(folder_id)
            if not date_folder_id:
                raise Exception(f"Could not create/get date folder in {folder_type}")
            print(f"✓ Created/Found date folder: {date_folder_id}")
            
            # Test listing images
            images = handler.list_images(date_folder_id)
            print(f"✓ Found {len(images)} images in {folder_type} folder")
        
        print("\n✅ All Drive handler tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Drive handler test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_drive_handler()


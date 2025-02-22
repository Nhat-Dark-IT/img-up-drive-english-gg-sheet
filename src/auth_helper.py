# src/auth_helper.py
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()

def get_google_service(service_name, version='v3'):
    """Create and return Google service object."""
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        scopes=[
            'https://www.googleapis.com/auth/drive',  # Full Drive access
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive.metadata',
            'https://www.googleapis.com/auth/spreadsheets'
        ]
    )
    
    return build(service_name, version, credentials=credentials)

def test_connection():
    """Test connection to Google services."""
    try:
        drive_service = get_google_service('drive')
        sheets_service = get_google_service('sheets', 'v4')
        
        # Test Drive API
        files = drive_service.files().list(
            q=f"'{os.getenv('GOOGLE_DRIVE_FOLDER_ID')}' in parents",
            pageSize=1
        ).execute()
        
        # Test Sheets API
        sheets_service.spreadsheets().get(
            spreadsheetId=os.getenv('GOOGLE_SHEETS_ID')
        ).execute()
        
        print("✅ Successfully connected to Google Drive and Sheets!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()
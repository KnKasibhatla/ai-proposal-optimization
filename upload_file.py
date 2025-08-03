#!/usr/bin/env python3
"""
Simple file upload script - bypasses all browser issues
"""

import requests
import sys
import os

def upload_file(file_path):
    """Upload a file to the backend server"""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        return False
    
    print(f"📁 Uploading: {file_path}")
    print(f"📊 File size: {os.path.getsize(file_path)} bytes")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/api/upload', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"📈 Records processed: {data.get('total_records', 'N/A')}")
            print(f"🎯 Win rate: {data.get('win_rate', 'N/A'):.1%}")
            print(f"👥 Unique clients: {data.get('unique_clients', 'N/A')}")
            print("\n🚀 AI model training completed!")
            return True
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server at http://localhost:5000")
        print("Make sure the backend server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_file.py <path_to_csv_file>")
        print("\nExample:")
        print("  python upload_file.py test_data.csv")
        print("  python upload_file.py ~/Desktop/manufacturing_data.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = upload_file(file_path)
    
    if success:
        print("\n🎉 Your data is now uploaded and the AI model is trained!")
        print("You can now use the web interface for predictions.")
    else:
        print("\n💡 Try these solutions:")
        print("1. Make sure the backend server is running (python app.py)")
        print("2. Check that the file path is correct")
        print("3. Ensure the file is a valid CSV format") 
# 1. Thêm vào đầu file để tắt logging
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt hoàn toàn TF logging
logging.getLogger('absl').setLevel(logging.ERROR)  # Chỉ hiện error cho absl
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import difflib
from spellchecker import SpellChecker
from vncorenlp import VnCoreNLP
from symspellpy import SymSpell, Verbosity
from importlib import resources  # Replace pkg_resources
import google.generativeai as genai
from google.api_core import retry
import os
from dotenv import load_dotenv

import time


# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Changed from append to insert

try:
    from config.settings import TESSERACT_PATH, IMAGE_PROCESS
except ImportError:
    print("Checking paths:")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    raise

class ImageProcessor:
    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        self.min_confidence = IMAGE_PROCESS['min_confidence']
        self.resize_width = IMAGE_PROCESS['resize_width']
        self.blur_kernel = IMAGE_PROCESS['blur_kernel']
        self.contrast_limit = IMAGE_PROCESS['contrast_limit']
        self.setup_symspell()
        self.spell_en = SpellChecker(language='en')
        try:
            self.vn_nlp = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx2g')
        except:
            print("Warning: VnCoreNLP not initialized")
            self.vn_nlp = None
        # Setup Gemini API
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY_GEMINI"))
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config
        )
        
    def enhance_with_gemini(self, text, output_path=None):
        """Sử dụng Gemini để cải thiện và chuẩn hóa văn bản."""
        prompt = f"""
        Phân tích và chuẩn hóa danh sách từ vựng sau đây:

        Văn bản gốc:
        {text}

        Yêu cầu:
        1. Xử lý chủ đề và trùng lặp:
        - Xác định và giữ nguyên chủ đề chính (VD: TOEIC, Fruits, etc.)
        - Loại bỏ hoàn toàn từ trùng lặp
        - Nếu trùng từ nhưng khác nghĩa -> gộp nghĩa
        - Với từ dính liền -> tách thành từ riêng (VD: "watermelonrind" -> "watermelon rind")

        2. Xử lý từ vựng:
        - Giữ nguyên cấu trúc "word = synonym" nếu có
        - Xóa từ có ký tự đặc biệt (VD: #, @, !, etc.)
        - Loại bỏ từ không hợp lệ hoặc không có nghĩa
        - Sửa lỗi chính tả tiếng Anh
        - Chuẩn hóa format

        3. Xử lý nghĩa:
        - Giữ nguyên nghĩa gốc đã có
        - Bổ sung nghĩa còn thiếu dựa trên nghĩa đã có
        - Sửa lỗi chính tả và ngữ pháp
        - Đảm bảo có đủ cả nghĩa Việt và Anh

        Định dạng đầu ra:
        Chủ đề | Từ vựng | Nghĩa tiếng Việt | Nghĩa tiếng Anh

        Ví dụ xử lý đúng:
        Input: 
        Fruits | watermelonrind | vỏ dưa hấu
        Fruits | watermelonrind | | the outer skin
        Output:
        Fruits | watermelon rind | vỏ dưa hấu | the outer skin of watermelon

        Input:
        TOEIC | look back on = remember | nhìn lại | think about past
        TOEIC | look back on | hồi tưởng | remember past
        Output:
        TOEIC | look back on = remember | nhìn lại, hồi tưởng | to think about something in the past

        Ví dụ loại bỏ:
        - Từ không hợp lệ: "Awabs", "ab#c", "!@#$"
        - Từ trùng lặp: "lemon zest" xuất hiện nhiều lần
        - Từ thiếu nghĩa hoàn toàn: "word | | |"

        Lưu ý quan trọng:
        1. PHẢI giữ nguyên chủ đề gốc
        2. PHẢI xóa hoàn toàn từ trùng lặp
        3. PHẢI tách từ dính liền
        4. PHẢI có đủ nghĩa tiếng Việt (Có dấu câu) và tiếng Anh
        5. KHÔNG thêm nghĩa mới nếu đã có nghĩa
        6. KHÔNG giữ từ có ký tự đặc biệt
        7. KHÔNG giữ từ không hợp lệ/không có nghĩa
        8. Với từ có "=", giữ nguyên cấu trúc
        9. KHÔNG giữ từ có ký tự đặc biệt
7. Nếu thiếu nghĩa -> Bổ sung dựa trên nghĩa đã có
        """



        try:
            # Gửi yêu cầu và nhận phản hồi từ Gemini
            time.sleep(10)
            chat = self.model.start_chat()
            response = chat.send_message(prompt)

            # Phân tích phản hồi từ Gemini
            enhanced_text = self.parse_gemini_response(response.text)
            
            # Lưu kết quả nếu có đường dẫn
            if output_path:
                # Đảm bảo output_path không phải None và kết quả trả về hợp lệ
                enhanced_path = output_path.replace('.jpg', '_enhanced.txt')
                
                # Lưu kết quả đã được cải thiện vào file
                with open(enhanced_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_text)

                print(f"Enhanced results saved to: {enhanced_path}")
            
            return enhanced_text

        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return text

    def parse_gemini_response(self, response):
        """Chuyển đổi phản hồi từ Gemini thành định dạng chuẩn."""
        lines = response.split('\n')
        formatted = []
        
        # Mở rộng skip_patterns để bắt tất cả header và số thứ tự
        skip_patterns = [
            # Headers patterns
            r'^(STT|Số\s*(TT|thứ\s*tự))',
            r'tiêu\s*đề.*từ\s*vựng.*ipa.*nghĩa.*',
            r'[-—]+\s*[-—]+\s*[-—]+\s*[-—]+\s*[-—]+',
            r'\*\*[^*]+\*\*\t\*\*[^*]+\*\*',
            r'^\d+\.\s+',  # Số thứ tự đầu dòng
            r'^[""]?[\d.]+[\s)""]', # Số có dấu ngoặc kép hoặc chấm
            r'Từ\s+vựng\s+về\s+.*', # Headers bắt đầu bằng "Từ vựng về"
        ]
    
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Skip headers và số thứ tự
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break
            if should_skip:
                continue
                
            # Loại bỏ số thứ tự ở đầu dòng nếu còn
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Parse fields
            parts = line.split('|')
            if len(parts) >= 4:  # Thay đổi từ 6 xuống 4 vì bỏ IPA
                title = parts[0].strip()
                word = parts[1].strip()
                vn_meaning = parts[2].strip()
                en_meaning = parts[3].strip()
                    
                # Remove ** markers
                title = re.sub(r'\*\*', '', title)
                word = re.sub(r'\*\*', '', word)
                
                formatted_line = f"{title}\t{word}\t{vn_meaning}\t{en_meaning}"
                formatted.append(formatted_line)
        
        return '\n'.join(formatted)

    def setup_symspell(self):
        """Setup SymSpell with new importlib.resources."""
        try:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Use importlib.resources instead of pkg_resources
            with resources.path('symspellpy', 'frequency_dictionary_en_82_765.txt') as dict_path:
                self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
        except Exception as e:
            print(f"SymSpell initialization failed: {e}")
            self.sym_spell = None
    def fix_spelling(self, text):
        """Fix spelling for English words only."""
        words = text.split()
        corrected = []
        
        for word in words:
            # Skip Vietnamese words (containing diacritics)
            if any(c in word for c in 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'):
                corrected.append(word)
                continue
            
            # Only spell check English words
            if word.isascii() and not self.spell_en.known([word]):
                correction = self.spell_en.correction(word)
                if correction:
                    word = correction
            
            corrected.append(word)
        
        return ' '.join(corrected)
            
    def load_image(self, image_path):
        """Load and resize image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Resize while maintaining aspect ratio
        aspect = image.shape[1] / float(image.shape[0])
        height = int(self.resize_width / aspect)
        return cv2.resize(image, (self.resize_width, height))
    def preprocess_image(self, image):
        """Preprocess image to handle any text/background color combination."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization (CLAHE) to improve contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Compute mean intensity to decide if background is light or dark
        mean_intensity = np.mean(enhanced)
        
        if mean_intensity > 127:
            # Light background, use normal binary threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Dark background, invert colors and apply threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use morphological transformations to enhance text structure
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return processed

    def remove_watermark(self, image):
        """Remove watermarks using inpainting."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get watermark mask
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Apply inpainting
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    def fix_spelling_with_symspell(self, text):
        """Fix English spelling using SymSpell."""
        words = text.split()
        corrected = []
        
        for word in words:
            # Skip Vietnamese words
            if any(c in word for c in 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'):
                corrected.append(word)
                continue
            
            # Check if word needs correction
            if word.isascii():
                suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    best_match = suggestions[0].term
                    # Add part of speech if in vocabulary
                    if best_match in self.vocab_pos:
                        word = f"{best_match} {self.vocab_pos[best_match]}"
                    else:
                        word = best_match
                        
            corrected.append(word)
        
        return ' '.join(corrected)
    def enhance_contrast(self, image):
        """Enhance image contrast."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.contrast_limit)
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # src/image_processor.py
    


    def clean_ocr_text(self, text):
        """Clean OCR output using regex, dictionary, and AI."""
        
        # 1. OCR common error dictionary
        ocr_fixes = {

        }

        # 2. Regular expression patterns
        patterns = {
            'numbered_line': r'(\d+)[.,]?\s*(.*)',
            'word_definition': r'([A-Za-z-]+)\s*=\s*([^=\n]+)',
            'example': r'E\.?g\.?\s*(.+)',
            'parentheses': r'\((.*?)\)',
        }

        # 3. Xử lý từng dòng
        lines = []
        for line in text.split('\n'):
            if not line.strip():
                continue

            # Dùng dictionary để sửa lỗi OCR
            for wrong, right in ocr_fixes.items():
                line = line.replace(wrong, right)

            # Dùng difflib để sửa từ sai chính tả
            words = line.split()
            for i, word in enumerate(words):
                closest_match = difflib.get_close_matches(word, ocr_fixes.keys(), n=1, cutoff=0.8)
                if closest_match:
                    words[i] = ocr_fixes[closest_match[0]]

            cleaned = " ".join(words)

            # Apply regex cleaning
            cleaned = re.sub(r'[^\w\s\(\)=,\.-]', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            lines.append(cleaned)

        return '\n'.join(lines)
    def clean_text(self, text):
        """Clean and correct OCR output."""
        if self.vn_nlp:
            try:
                # Apply Vietnamese word segmentation
                sentences = self.vn_nlp.tokenize(text)
                text = " ".join([" ".join(sent) for sent in sentences])
            except:
                pass
                
        return text.strip()
    
    def ai_correct_text(self, text):
        """Apply AI-based text correction."""
        # Split into sentences
        sentences = text.split('.')
        corrected = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 1. Fix capitalization
            sentence = sentence.strip().capitalize()
            
            # 2. Fix Vietnamese diacritics
            sentence = self.fix_vietnamese_diacritics(sentence)
            
            # 3. Fix grammar structure
            sentence = self.fix_grammar_structure(sentence)
            
            corrected.append(sentence)
        
        return '. '.join(corrected)
    
    def fix_vietnamese_diacritics(self, text):
        """Fix missing or incorrect Vietnamese diacritics."""
        diacritic_map = {
            'muon': 'muốn',
            'nguoi': 'người',
            'hoc': 'học',
            'tieng': 'tiếng',
            # Add more mappings
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in diacritic_map:
                words[i] = diacritic_map[word.lower()]
        
        return ' '.join(words)
    
    def fix_grammar_structure(self, text):
        """Fix common grammar patterns."""
        # Example patterns
        patterns = [
            (r'(\w+)\s*=\s*([^=]+)', r'\1 = \2'),  # Fix equals spacing
            (r'eg\.,?', 'E.g.'),  # Fix example markers
            (r'\(\s*([^)]+)\s*\)', r'(\1)'),  # Fix parentheses spacing
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    def clean_numbering_and_headers(self, text):
        """Loại bỏ số thứ tự và headers."""
        if not text:
            return text
            
        # Patterns cần loại bỏ
        patterns = [
            # Headers
            r'^STT\s*Tiêu\s*đề\s*Từ\s*Vựng\s*IPA\s*Nghĩa.*$',
            r'^[-—]+(\s+[-—]+)*$',
            # Số thứ tự
            r'^\d+\.\s*',
            r'^[""]?\d+\.?\s*[""]?',
            # Các cột không cần thiết
            r'STT\s*',
            r'Tiêu\s*đề\s*',
            r'Từ\s*Vựng\s*',
            r'IPA\s*',
            r'Nghĩa\s*Tiếng\s*Việt\s*',
            r'Nghĩa\s*tiếng\s*anh\s*'
        ]
        
        # Xử lý từng dòng
        lines = []
        for line in text.split('\n'):
            if not line.strip():
                continue
                
            # Loại bỏ các pattern
            line = line.strip()
            for pattern in patterns:
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
                
            if line.strip():
                lines.append(line.strip())
                
        return '\n'.join(lines)
    def process_image(self, image_path, output_path=None):
        """Process image with enhanced error handling and Gemini integration."""
        try:
            print(f"\nProcessing image: {image_path}")
            
            # 1. Load and preprocess image
            image = self.load_image(image_path)
            print("Image loaded successfully")
            
            preprocessed = self.preprocess_image(image)
            print("Preprocessing completed")
            
            if output_path:
                cv2.imwrite(output_path, preprocessed)
                print(f"Preprocessed image saved to: {output_path}")
            
            # 2. Run OCR with config
            custom_config = (
                '--oem 3 '
                '--psm 6 '
                '-l eng+vie '
                '--dpi 300 '
                '-c preserve_interword_spaces=1 '
                '-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(),.:-= '
            )
            
            pil_image = Image.fromarray(preprocessed)
            text = pytesseract.image_to_string(
                pil_image,
                lang='eng+vie',
                config=custom_config
            )
            print("OCR completed")
            
            if not text:
                print("⚠️ Warning: OCR produced empty result")
                return None
            
            # 3. Clean OCR text
            cleaned_text = self.clean_text(text)
            cleaned_text = self.clean_numbering_and_headers(cleaned_text)
            result = self.format_vocabulary(cleaned_text)
            print("Basic text cleaning completed")
            # 4. Enhance with Gemini AI
            enhanced_text = self.enhance_with_gemini(cleaned_text)
            print("AI enhancement completed")
            # 5. Clean lại lần nữa sau khi qua AI
            enhanced_text = self.clean_numbering_and_headers(enhanced_text)
            print("Text formatting completed")
            # 5. Format vocabulary
            print("Text formatting completed")
            
            # 6. Save results
            if output_path:
                # Save raw OCR result
                ocr_path = output_path.replace('.jpg', '_ocr.txt')
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                    
                # Save enhanced result
                enhanced_path = output_path.replace('.jpg', '_enhanced.txt')
                with open(enhanced_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_text)
                
                print(f"Results saved to: {enhanced_path}")
            
            return enhanced_text
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None  # Trả về None thay vì raise Exception để tránh ảnh hưởng đến luồng xử lý chính
        
    def format_vocabulary(self, text):
        """Format vocabulary entries consistently."""
        lines = text.split('\n')
        formatted = []
        current_number = 0
        
        # Enhanced pattern matching for vocabulary entries
        vocab_pattern = r'''(?:(\d+)\.?\s*)?  # Optional number
                           ([a-zA-Z\s-]+)\s*   # English word/phrase
                           \(?([a-zA-z/]+)\)?  # Part of speech
                           \s*:?\s*            # Optional colon
                           (.+)                # Definition'''
        
        for line in lines:
            if not line.strip():
                continue
            
            # Check for STUDY header
            
            # Try to match vocabulary entry
            match = re.match(vocab_pattern, line, re.VERBOSE)
            if match:
                _, word, pos, definition = match.groups()
                
                # Clean up components
                word = word.strip().lower()
                pos = pos.strip().lower()
                definition = definition.strip()
                
                # Fix common OCR errors
                fixes = {
        
                }
                
                for wrong, right in fixes.items():
                    pos = pos.replace(wrong, right)
                    definition = definition.replace(wrong, right)
                
                current_number += 1
                formatted_line = f"{current_number}. {word} ({pos}): {definition}"
                formatted.append(formatted_line)
                
        return '\n'.join(formatted)
def test_image_processor():
    """Test image processing functionality."""
    try:
        # Initialize processor
        print("\nInitializing Image Processor...")
        processor = ImageProcessor()
        
        # Đường dẫn tới ảnh test
        test_dir = os.path.join(project_root, "tests", "images")
        test_image = os.path.join(test_dir, "sample13.jpg")
        output_dir = os.path.join(test_dir, "output")
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(test_image):
            raise Exception(f"Test image not found at: {test_image}")
            
        # Process image
        output_path = os.path.join(output_dir, "processed_sample13.jpg")
        text = processor.process_image(test_image, output_path)
        
        if not text:
            raise Exception("OCR returned no text")
            
        print("\n✅ Image processing test passed!")
        print(f"Input image: {test_image}")
        print(f"Output image: {output_path}")
        print(f"Extracted text:\n{text}")
        return True
        
    except Exception as e:
        print(f"\n❌ Image processing test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_image_processor()
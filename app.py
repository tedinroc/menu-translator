from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import base64
import io
import re
import logging
import sys

# 設置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 添加控制台處理器
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 檢查 API key
if not os.getenv('OPENAI_API_KEY'):
    logger.error("OpenAI API key not found in environment variables!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        logger.info("Starting translation request")
        
        # 檢查請求數據
        if not request.json:
            logger.error("No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Get the image data from the request
        image_data = request.json.get('image')
        if not image_data:
            logger.error("No image data in request")
            return jsonify({'error': 'No image data provided'}), 400
            
        target_language = request.json.get('target_language', 'English')
        logger.info(f"Target language: {target_language}")
        
        try:
            # Convert base64 to image
            logger.debug("Processing image data")
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Validate image
            logger.debug(f"Image size: {image.size}")
            logger.debug(f"Image mode: {image.mode}")
            logger.debug(f"Image format: {image.format}")
            
            # Check image dimensions
            if image.size[0] < 100 or image.size[1] < 100:
                logger.error("Image too small")
                return jsonify({'error': 'Image too small. Please provide a larger image.'}), 400
                
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                logger.debug("Converting RGBA to RGB")
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                logger.debug(f"Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Resize if image is too large
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                logger.debug(f"Resizing image from {image.size} to {max_size}")
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Enhance image quality
            from PIL import ImageEnhance
            
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Adjust sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.3)  # Increase sharpness by 30%
            
            # Convert image to base64 for OpenAI API
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Log image size
            logger.debug(f"Processed image size: {len(img_str)} bytes")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400

        try:
            logger.info("Calling OpenAI API")
            # Call OpenAI API for translation using GPT-4O
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a professional menu translator. Your task is to:
                            1. Carefully examine the menu image
                            2. Extract all visible text items
                            3. Translate each item accurately
                            4. Format each translation as:
                               [Original Text] -> [Translation]
                            
                            Important rules:
                            - Only translate text you can clearly see
                            - Keep numbers and special characters unchanged
                            - Skip any text you're unsure about
                            - If you can't see any text clearly, explain why
                            - If the image is unclear, suggest how to improve it
                            
                            Example format:
                            Sushi Roll -> 寿司卷
                            Miso Soup -> 味噌汤
                            """
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Please examine this menu image carefully and translate all visible text to {target_language}. If you can't read the text clearly, please explain why and suggest improvements."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg:
                    logger.error("Rate limit exceeded")
                    return jsonify({
                        'error': 'Service is temporarily unavailable due to high demand. Please try again in a few minutes.',
                        'error_type': 'rate_limit'
                    }), 429
                elif 'quota' in error_msg:
                    logger.error("API quota exceeded")
                    return jsonify({
                        'error': 'Daily translation limit reached. Please try again tomorrow.',
                        'error_type': 'quota_exceeded'
                    }), 429
                else:
                    logger.error(f"OpenAI API error: {str(e)}")
                    return jsonify({
                        'error': 'An error occurred while processing your request. Please try again.',
                        'error_type': 'api_error'
                    }), 500
            logger.info("OpenAI API call successful")
            
            # Log the raw response
            logger.debug(f"Raw API response: {response}")
            
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return jsonify({'error': f'Error processing API response: {str(e)}'}), 500

        try:
            # Get the translation result
            translation = response.choices[0].message.content
            logger.debug(f"Raw translation: {translation}")

            # Process the translation to ensure consistent format
            # Split into lines and filter out empty lines
            lines = [line.strip() for line in translation.split('\n') if line.strip()]
            logger.debug(f"Split lines: {lines}")
            
            # Handle Chinese variants
            target_language = request.json.get('target_language', 'English')
            if target_language in ['Traditional Chinese', 'Simplified Chinese']:
                try:
                    from opencc import OpenCC
                    cc = OpenCC('s2t' if target_language == 'Traditional Chinese' else 't2s')
                    lines = [cc.convert(line) for line in lines]
                    logger.debug(f"Converted Chinese text: {lines}")
                except ImportError:
                    logger.warning("OpenCC not installed, skipping Chinese conversion")
                except Exception as e:
                    logger.error(f"Error converting Chinese text: {str(e)}")
            
            # Process each line to ensure it has the correct format
            formatted_lines = []
            for line in lines:
                logger.debug(f"Processing line: {line}")
                
                # Skip lines that look like instructions or notes
                if any(word in line.lower() for word in ['note:', 'translation:', 'original:', 'menu:', 'item:']):
                    logger.debug(f"Skipping instruction line: {line}")
                    continue

                # Try to extract original and translated text using various patterns
                translation_pair = None
                
                # Try different separators
                for separator in [' -> ', ': ', ' - ', '→', '>']:
                    if separator in line:
                        parts = line.split(separator, 1)
                        if len(parts) == 2:
                            translation_pair = parts
                            break

                # If no separator found, try to find any two text parts separated by space
                if not translation_pair:
                    # Remove common list markers
                    clean_line = re.sub(r'^[\d\.\-\*\•\★\⭐]+\s*', '', line)
                    parts = clean_line.split('  ', 1)  # Split by double space
                    if len(parts) == 2:
                        translation_pair = parts

                if translation_pair:
                    original, translated = translation_pair
                    original = original.strip()
                    translated = translated.strip()
                    
                    # Remove common list markers and brackets
                    original = re.sub(r'^[\d\.\-\*\•\★\⭐\[\(\{]+\s*', '', original)
                    original = re.sub(r'\s*[\]\)\}]+$', '', original)
                    translated = re.sub(r'^[\d\.\-\*\•\★\⭐\[\(\{]+\s*', '', translated)
                    translated = re.sub(r'\s*[\]\)\}]+$', '', translated)
                    
                    if original and translated:
                        formatted_line = f"{original}: {translated}"
                        logger.debug(f"Added formatted line: {formatted_line}")
                        formatted_lines.append(formatted_line)
                else:
                    logger.debug(f"Could not parse line: {line}")

            # Join the formatted lines back together
            formatted_translation = '\n'.join(formatted_lines)
            logger.debug(f"Final formatted translation: {formatted_translation}")
            
            if not formatted_lines:
                logger.warning("No valid translations found after formatting")
                # Instead of returning error, return the raw translation
                return jsonify({'translation': translation})
            
            return jsonify({'translation': formatted_translation})
            
        except Exception as e:
            logger.error(f"Error processing translation: {str(e)}")
            return jsonify({'error': f'Error processing translation: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

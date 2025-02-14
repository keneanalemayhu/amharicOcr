import numpy as np # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance # type: ignore
import os
from pathlib import Path
import random
import glob

class AmharicDatasetGenerator:
    def __init__(self, output_dir="dataset"):
        # Define character families
        self.character_families = {
            'ሀ': ['ሀ', 'ሁ', 'ሂ', 'ሃ', 'ሄ', 'ህ', 'ሆ'],
            'ለ': ['ለ', 'ሉ', 'ሊ', 'ላ', 'ሌ', 'ል', 'ሎ', 'ሏ'],
            'ሐ': ['ሐ', 'ሑ', 'ሒ', 'ሓ', 'ሔ', 'ሕ', 'ሖ', 'ሗ'],
            'መ': ['መ', 'ሙ', 'ሚ', 'ማ', 'ሜ', 'ም', 'ሞ', 'ሟ'],
            'ሠ': ['ሠ', 'ሡ', 'ሢ', 'ሣ', 'ሤ', 'ሥ', 'ሦ', 'ሧ'],
            'ረ': ['ረ', 'ሩ', 'ሪ', 'ራ', 'ሬ', 'ር', 'ሮ', 'ሯ'],
            'ሰ': ['ሰ', 'ሱ', 'ሲ', 'ሳ', 'ሴ', 'ስ', 'ሶ', 'ሷ'],
            'ሸ': ['ሸ', 'ሹ', 'ሺ', 'ሻ', 'ሼ', 'ሽ', 'ሾ', 'ሿ'],
            'ቀ': ['ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ', 'ቋ'],
            'በ': ['በ', 'ቡ', 'ቢ', 'ባ', 'ቤ', 'ብ', 'ቦ', 'ቧ'],
            'ቨ': ['ቨ', 'ቩ', 'ቪ', 'ቫ', 'ቬ', 'ቭ', 'ቮ', 'ቯ'],
            'ተ': ['ተ', 'ቱ', 'ቲ', 'ታ', 'ቴ', 'ት', 'ቶ', 'ቷ'],
            'ቸ': ['ቸ', 'ቹ', 'ቺ', 'ቻ', 'ቼ', 'ች', 'ቾ', 'ቿ'],
            'ኀ': ['ኀ', 'ኁ', 'ኂ', 'ኃ', 'ኄ', 'ኅ', 'ኆ', 'ኋ'],
            'ነ': ['ነ', 'ኑ', 'ኒ', 'ና', 'ኔ', 'ን', 'ኖ', 'ኗ'],
            'ኘ': ['ኘ', 'ኙ', 'ኚ', 'ኛ', 'ኜ', 'ኝ', 'ኞ', 'ኟ'],
            'አ': ['አ', 'ኡ', 'ኢ', 'ኣ', 'ኤ', 'እ', 'ኦ'],
            'ከ': ['ከ', 'ኩ', 'ኪ', 'ካ', 'ኬ', 'ክ', 'ኮ', 'ኳ'],
            'ኸ': ['ኸ', 'ኹ', 'ኺ', 'ኻ', 'ኼ', 'ኽ', 'ኾ', 'ዃ'],
            'ወ': ['ወ', 'ዉ', 'ዊ', 'ዋ', 'ዌ', 'ው', 'ዎ'],
            'ዐ': ['ዐ', 'ዑ', 'ዒ', 'ዓ', 'ዔ', 'ዕ', 'ዖ'],
            'ዘ': ['ዘ', 'ዙ', 'ዚ', 'ዛ', 'ዜ', 'ዝ', 'ዞ', 'ዟ'],
            'ዠ': ['ዠ', 'ዡ', 'ዢ', 'ዣ', 'ዤ', 'ዥ', 'ዦ', 'ዧ'],
            'የ': ['የ', 'ዩ', 'ዪ', 'ያ', 'ዬ', 'ይ', 'ዮ'],
            'ደ': ['ደ', 'ዱ', 'ዲ', 'ዳ', 'ዴ', 'ድ', 'ዶ', 'ዷ'],
            'ጀ': ['ጀ', 'ጁ', 'ጂ', 'ጃ', 'ጄ', 'ጅ', 'ጆ', 'ጇ'],
            'ገ': ['ገ', 'ጉ', 'ጊ', 'ጋ', 'ጌ', 'ግ', 'ጎ', 'ጓ'],
            'ጠ': ['ጠ', 'ጡ', 'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ', 'ጧ'],
            'ጨ': ['ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ', 'ጯ'],
            'ጰ': ['ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ', 'ጵ', 'ጶ', 'ጷ'],
            'ጸ': ['ጸ', 'ጹ', 'ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ', 'ጿ'],
            'ፀ': ['ፀ', 'ፁ', 'ፂ', 'ፃ', 'ፄ', 'ፅ', 'ፆ'],
            'ፈ': ['ፈ', 'ፉ', 'ፊ', 'ፋ', 'ፌ', 'ፍ', 'ፎ', 'ፏ'],
            'ፐ': ['ፐ', 'ፑ', 'ፒ', 'ፓ', 'ፔ', 'ፕ', 'ፖ', 'ፗ'],

            # Amharic Numerals
            '፩': ['፩', '1'],
            '፪': ['፪', '2'],
            '፫': ['፫', '3'],
            '፬': ['፬', '4'],
            '፭': ['፭', '5'],
            '፮': ['፮', '6'],
            '፯': ['፯', '7'],
            '፰': ['፰', '8'],
            '፱': ['፱', '9'],
            '፲': ['፲', '10']
        }
        
        # Get project root directory (2 levels up from this file)
        self.project_root = Path(__file__).resolve().parents[2]
        
        # Get project root and set paths
        self.output_dir = Path(__file__).resolve().parents[2] / output_dir
        self.fonts_dir = Path(__file__).resolve().parents[2] / 'fonts'

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Flatten character families into a list of unique characters
        self.characters = []
        for family_chars in self.character_families.values():
            self.characters.extend(family_chars)
        self.characters = list(set(self.characters))

        # Load fonts
        self.font_paths = list(self.fonts_dir.glob('*.ttf')) + list(self.fonts_dir.glob('*.otf'))
        if not self.font_paths:
            raise Exception(f"No .ttf or .otf fonts found in: {self.fonts_dir}")

        print(f"Found {len(self.font_paths)} fonts in: {self.fonts_dir}")
        for font in self.font_paths:
            print(f"- {font.name}")

        # Settings
        self.image_size = (96, 96)
        self.font_sizes = range(40, 60, 4)
        self.train_ratio = 0.8
        self.vertical_padding = 0.25
        self.horizontal_padding = 0.25

    def get_random_font(self, size):
        """Get a random font from available fonts"""
        font_path = random.choice(self.font_paths)
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"Warning: Could not load font {font_path}: {str(e)}")
            # Fallback to the first font that works
            for backup_font in self.font_paths:
                try:
                    return ImageFont.truetype(backup_font, size)
                except:
                    continue
            raise Exception("None of the fonts could be loaded!")

    def calculate_safe_position(self, draw, character, font, image_size):
        """Calculate position ensuring character is perfectly centered"""
        # Get character bounding box
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate available space
        available_width = image_size[0] * (1 - 2 * self.horizontal_padding)
        available_height = image_size[1] * (1 - 2 * self.vertical_padding)
        
        # Scale factor if text is too large
        scale_x = available_width / text_width if text_width > available_width else 1
        scale_y = available_height / text_height if text_height > available_height else 1
        scale = min(scale_x, scale_y)
        
        # Calculate true center position
        x = (image_size[0] - text_width * scale) / 2
        y = (image_size[1] - text_height * scale) / 2
        
        # Fine-tuning vertical position based on character type
        if any(u_char in character for u_char in ['ሁ', 'ሉ', 'ሙ', 'ሡ', 'ሩ', 'ሱ', 'ሹ', 'ቁ', 'ቑ', 'ቡ', 'ቱ', 'ቹ']):
            y = y - (text_height * 0.15)  # Move up more for 'U' forms
            
        return x, y, scale

    def generate_character_image(self, character, font_size, noise_level=0.1):
        # Create new image with white background
        image = Image.new('L', self.image_size, color=255)
        draw = ImageDraw.Draw(image)
        
        # Get random font
        font = self.get_random_font(font_size)
        
        # Calculate position ensuring character is fully visible and centered
        x, y, scale = self.calculate_safe_position(draw, character, font, self.image_size)
        
        # If scaling is needed, adjust font size
        if scale < 1:
            font_size = int(font_size * scale)
            font = self.get_random_font(font_size)
            x, y, _ = self.calculate_safe_position(draw, character, font, self.image_size)
        
        # Draw character
        draw.text((x, y), character, font=font, fill=0)
        
        # Add random noise
        if random.random() > 0.5:
            img_array = np.array(image)
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return self.apply_augmentation(image)

    def apply_augmentation(self, image):
        """Apply various augmentation techniques to the image"""
        # Convert to RGB for some transformations
        image = image.convert('RGB')
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.5, 2.0)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        # Random blur
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
        
        # Random rotation
        if random.random() > 0.5:
            rotation = random.uniform(-15, 15)
            image = image.rotate(rotation, expand=False, fillcolor=(255, 255, 255))
        
        # Convert back to grayscale
        image = image.convert('L')
        
        return image

    def count_existing_images(self, char_dir):
        """Count existing images in a directory"""
        if not char_dir.exists():
            return 0
        return len(list(char_dir.glob('*.png')))

    def generate_dataset(self, samples_per_char=100):
        # Create train and validation directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        failed_generations = []
        total_new_images = 0
        
        print("\nInitial dataset statistics:")
        total_existing_train = sum(self.count_existing_images(train_dir / char) 
                                 for char in self.characters)
        total_existing_val = sum(self.count_existing_images(val_dir / char) 
                               for char in self.characters)
        print(f"Existing training images: {total_existing_train}")
        print(f"Existing validation images: {total_existing_val}")
        print(f"Total existing images: {total_existing_train + total_existing_val}")
        
         # Generate for each family and its characters
        for family, chars in self.character_families.items():
            print(f"\nProcessing family: {family}")
            
            # Create family directories
            train_family_dir = train_dir / family
            val_family_dir = val_dir / family
            train_family_dir.mkdir(exist_ok=True)
            val_family_dir.mkdir(exist_ok=True)
            
            for char in chars:
                print(f"Generating samples for character: {char}")
                
                # Create character directories
                train_char_dir = train_family_dir / char
                val_char_dir = val_family_dir / char
                train_char_dir.mkdir(exist_ok=True)
                val_char_dir.mkdir(exist_ok=True)
                
                # Generate images for this character
                for i in range(samples_per_char):
                    try:
                        font_size = random.choice(self.font_sizes)
                        img = self.generate_character_image(char, font_size)
                        
                        # Decide whether to put in training or validation set
                        if random.random() < self.train_ratio:
                            save_dir = train_char_dir
                        else:
                            save_dir = val_char_dir
                        
                        # Generate unique filename
                        base_count = len(list(save_dir.glob('*.png')))
                        img.save(save_dir / f"{char}_{base_count:04d}.png")
                        total_new_images += 1
                        
                    except Exception as e:
                        failed_generations.append((char, str(e)))
                        print(f"Warning: Failed to generate image for character {char}: {str(e)}")
                        continue
        
        print("\nDataset generation complete!")
        
        # Print final dataset statistics
        final_train = sum(self.count_existing_images(train_dir / char) 
                         for char in self.characters)
        final_val = sum(self.count_existing_images(val_dir / char) 
                       for char in self.characters)
        
        print(f"\nFinal Dataset Statistics:")
        print(f"Total characters: {len(self.characters)}")
        print(f"New images generated: {total_new_images}")
        print(f"Total training images: {final_train}")
        print(f"Total validation images: {final_val}")
        print(f"Total images: {final_train + final_val}")
        
        if failed_generations:
            print("\nWarning: Some characters failed to generate:")
            for char, error in failed_generations:
                print(f"- Character '{char}': {error}")

# Usage example
if __name__ == "__main__":
    generator = AmharicDatasetGenerator("amharic_dataset")
    generator.generate_dataset(samples_per_char=100)
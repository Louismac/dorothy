# generate_font_atlas.py
import json
from PIL import Image, ImageDraw, ImageFont

def generate_font_atlas(font_path, output_name, size=32, charset=None):
    if charset is None:
        charset = ''.join(chr(i) for i in range(32, 127))
    
    try:
        font = ImageFont.truetype(font_path, size)
    except:
        print(f"Could not load font from {font_path}")
        font = ImageFont.load_default()
    
    atlas_size = 512
    padding = 2
    glyphs_per_row = 16
    cell_size = atlas_size // glyphs_per_row
    
    atlas = Image.new('L', (atlas_size, atlas_size), 0)
    draw = ImageDraw.Draw(atlas)
    
    dorothy_format = {
        'size': size,
        'atlas_size': [atlas_size, atlas_size],
        'glyphs': {}
    }
    
    for idx, char in enumerate(charset):
        row = idx // glyphs_per_row
        col = idx % glyphs_per_row
        
        if row >= glyphs_per_row:
            break
        
        x = col * cell_size + padding
        y = row * cell_size + padding
        
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0] if bbox else 0
        char_height = bbox[3] - bbox[1] if bbox else 0
        
        # Draw character
        if char.strip():
            draw.text((x - bbox[0], y - bbox[1]), char, font=font, fill=255)
        
        
        u0 = x / atlas_size
        v0 = y / atlas_size  
        u1 = (x + char_width) / atlas_size
        v1 = (y + char_height) / atlas_size
        
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0] if bbox else 0
        char_height = bbox[3] - bbox[1] if bbox else 0

        # Draw character
        if char.strip():
            draw.text((x - bbox[0], y - bbox[1]), char, font=font, fill=255)

        # Use bbox width as advance with small padding for spacing
        advance = char_width + 4  # Simple: width + 2px spacing
        
        dorothy_format['glyphs'][char] = {
            'uv': [u0, v0, u1, v1],
            'advance': advance,
            'offset': [bbox[0], bbox[1]],
            'size': [char_width, char_height]
        }
    
    atlas.save(f'{output_name}.png')
    
    with open(f'{output_name}.json', 'w') as f:
        json.dump(dorothy_format, f, indent=2)
    
    print(f"Generated {output_name}.png and {output_name}.json")

if __name__ == '__main__':
    import os
    
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        'C:\\Windows\\Fonts\\arial.ttf',
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            generate_font_atlas(path, 'font_atlas', size=32)
            break
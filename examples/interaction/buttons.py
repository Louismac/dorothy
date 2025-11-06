import numpy as np
from cv2 import circle
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        # Create a simple button
        def on_button_click(btn):
            print(f"Button '{btn.text}' was clicked!")
        
        def on_hover(btn):
            print(f"Button '{btn.text}' was hovered!")
        
        dot.create_button(300, 250, 200, 50, 
                        text="Click Me",
                        id="button1",
                        on_release=on_button_click, on_hover=on_hover)

    def draw(self):
        dot.background((40, 40, 50))
        
        # Update and draw buttons
        dot.update_buttons()
        dot.draw_buttons()

MySketch()
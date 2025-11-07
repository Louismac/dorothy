"""Button system for Dorothy"""
import numpy as np
from typing import Callable, Tuple, Optional

class Button:
    """Interactive button with callbacks"""
    
    def __init__(self, x, y, width, height, text="", id=None, 
                 on_press=None, on_hover=None, on_release=None):
        """Create a button
        
        Args:
            x, y: Position (top-left corner)
            width, height: Button dimensions
            text: Button label
            id: Optional identifier
            on_press: Callback when mouse pressed on button
            on_hover: Callback when mouse hovers over button
            on_release: Callback when mouse released on button
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.id = id
        
        # Callbacks
        self.on_press = on_press
        self.on_hover = on_hover
        self.on_release = on_release
        
        # State
        self.is_hovered = False
        self.is_pressed = False
        self.was_pressed_last_frame = False
        
        # Style
        self.idle_color = (200, 200, 200, 255)
        self.hover_color = (220, 220, 220, 255)
        self.pressed_color = (150, 150, 150, 255)
        self.text_color = (0, 0, 0, 255)
        self.border_color = (100, 100, 100, 255)
        self.border_width = 2
        self.corner_radius = 5
        
    def contains_point(self, px, py):
        """Check if point is inside button"""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def update(self, mouse_x, mouse_y, mouse_pressed):
        """Update button state and trigger callbacks
        
        Args:
            mouse_x, mouse_y: Current mouse position
            mouse_pressed: Whether mouse is currently pressed
        
        Returns:
            True if button was clicked (pressed and released)
        """
        # Check hover state
        was_hovered = self.is_hovered
        self.is_hovered = self.contains_point(mouse_x, mouse_y)
        
        # Trigger hover callback on enter
        if self.is_hovered and not was_hovered:
            if self.on_hover:
                self.on_hover(self)
        
        # Check press state
        clicked = False
        
        if self.is_hovered and mouse_pressed:
            # Mouse down on button
            if not self.was_pressed_last_frame:
                self.is_pressed = True
                if self.on_press:
                    self.on_press(self)
        elif self.is_pressed and not mouse_pressed:
            # Mouse released
            self.is_pressed = False
            if self.is_hovered:
                # Released on button = click
                clicked = True
                if self.on_release:
                    self.on_release(self)
        
        self.was_pressed_last_frame = mouse_pressed
        
        return clicked
    
    def draw(self, dot):
        """Draw the button using Dorothy
        
        Args:
            dot: Dorothy instance
        """
        # Determine color based on state
        if self.is_pressed:
            color = self.pressed_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.idle_color
        
        # Draw button background
        dot.fill(color)
        if self.border_width > 0:
            dot.stroke(self.border_color)
            dot.set_stroke_weight(self.border_width)
        else:
            dot.no_stroke()

        dot.rectangle((self.x, self.y), (self.x + self.width, self.y + self.height))
        
        # # Draw text (centered)
        # if self.text:
        #     # Calculate font size based on button height
        #     font_size = int(self.height * 0.2)  # 40% of button height
            
        #     # Center text
        #     text_x = self.x + self.width // 2
        #     text_y = self.y + self.height // 2 - font_size // 3  # Adjust for visual centering
            
        #     dot.text(self.text, (text_x, text_y), 
        #             font_size=font_size, 
        #             align='center',
        #             color=self.text_color)
    
    def _draw_rounded_rect(self, dot):
        """Draw a rounded rectangle (simplified)"""
        # For now, just draw regular rectangle
        # Full implementation would need arc drawing
        dot.rectangle((self.x, self.y), 
                     (self.x + self.width, self.y + self.height))
    
    def set_style(self, idle_color=None, hover_color=None, pressed_color=None,
                  text_color=None, border_color=None, border_width=None):
        """Set button visual style"""
        if idle_color: self.idle_color = idle_color
        if hover_color: self.hover_color = hover_color
        if pressed_color: self.pressed_color = pressed_color
        if text_color: self.text_color = text_color
        if border_color: self.border_color = border_color
        if border_width is not None: self.border_width = border_width


class ButtonManager:
    """Manages multiple buttons"""
    
    def __init__(self):
        self.buttons = []
        
    def add(self, button):
        """Add a button to the manager"""
        self.buttons.append(button)
        return button
    
    def create(self, x, y, width, height, text="", id=None,
               on_press=None, on_hover=None, on_release=None):
        """Create and add a button"""
        button = Button(x, y, width, height, text, id, 
                       on_press, on_hover, on_release)
        self.add(button)
        return button
    
    def update(self, mouse_x, mouse_y, mouse_pressed):
        """Update all buttons
        
        Returns:
            List of buttons that were clicked this frame
        """
        clicked = []
        for button in self.buttons:
            if button.update(mouse_x, mouse_y, mouse_pressed):
                clicked.append(button)
        return clicked
    
    def draw(self, dot):
        """Draw all buttons"""
        for button in self.buttons:
            button.draw(dot)
    
    def get_by_id(self, id):
        """Get button by ID"""
        for button in self.buttons:
            if button.id == id:
                return button
        return None
    
    def remove(self, button):
        """Remove a button"""
        if button in self.buttons:
            self.buttons.remove(button)
    
    def remove_by_id(self, id):
        """Remove button by ID"""
        button = self.get_by_id(id)
        if button:
            self.remove(button)
    
    def clear(self):
        """Remove all buttons"""
        self.buttons.clear()
"""Horizontal slider system for Dorothy"""
from typing import Callable, Optional


class Slider:
    """Interactive horizontal slider with callbacks"""

    def __init__(self, x, y, width, height, min_val=0.0, max_val=1.0,
                 value=None, label="", id=None,
                 on_change=None, on_press=None, on_hover=None, on_release=None):
        """Create a horizontal slider

        Args:
            x, y: Position (top-left corner of track)
            width, height: Slider dimensions
            min_val: Minimum value
            max_val: Maximum value
            value: Initial value (defaults to min_val)
            label: Optional text label
            id: Optional identifier
            on_change: Callback when value changes — receives (slider, value)
            on_press: Callback when handle is grabbed — receives (slider)
            on_hover: Callback when mouse enters slider area — receives (slider)
            on_release: Callback when handle is released — receives (slider)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = min_val if value is None else float(value)
        self.label = label
        self.id = id

        # Callbacks
        self.on_change = on_change
        self.on_press = on_press
        self.on_hover = on_hover
        self.on_release = on_release

        # State
        self.is_hovered = False
        self.is_dragging = False
        self.was_pressed_last_frame = False

        # Style
        self.track_color = (180, 180, 180, 255)
        self.track_fill_color = (100, 150, 220, 255)
        self.handle_color = (220, 220, 220, 255)
        self.handle_hover_color = (240, 240, 240, 255)
        self.handle_drag_color = (150, 150, 150, 255)
        self.label_color = (0, 0, 0, 255)
        self.border_color = (100, 100, 100, 255)
        self.border_width = 2
        self.handle_width = 12

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    def _normalised(self):
        """Return value normalised to [0, 1]"""
        span = self.max_val - self.min_val
        if span == 0:
            return 0.0
        return (self.value - self.min_val) / span

    def _handle_x(self):
        """Return the left edge x of the handle"""
        return self.x + self._normalised() * (self.width - self.handle_width)

    def _value_from_mouse(self, mx):
        """Convert a mouse x position to a clamped value"""
        rel = (mx - self.x - self.handle_width / 2) / (self.width - self.handle_width)
        rel = max(0.0, min(1.0, rel))
        return self.min_val + rel * (self.max_val - self.min_val)

    def set_value(self, value):
        """Programmatically set the value (clamped, fires on_change)"""
        new_val = max(self.min_val, min(self.max_val, value))
        if new_val != self.value:
            self.value = new_val
            if self.on_change:
                self.on_change(self, self.value)

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def contains_point(self, px, py):
        """Check if point is inside the slider track area"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def _handle_contains_point(self, px, py):
        """Check if point is inside the handle"""
        hx = self._handle_x()
        return (hx <= px <= hx + self.handle_width and
                self.y <= py <= self.y + self.height)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, mouse_x, mouse_y, mouse_pressed):
        """Update slider state and trigger callbacks

        Args:
            mouse_x, mouse_y: Current mouse position
            mouse_pressed: Whether mouse is currently pressed

        Returns:
            Current value of the slider
        """
        # Hover
        was_hovered = self.is_hovered
        self.is_hovered = self.contains_point(mouse_x, mouse_y)

        if self.is_hovered and not was_hovered:
            if self.on_hover:
                self.on_hover(self)

        # Begin drag when pressing handle or track
        if self.is_hovered and mouse_pressed and not self.was_pressed_last_frame:
            self.is_dragging = True
            if self.on_press:
                self.on_press(self)

        # Drag
        if self.is_dragging and mouse_pressed:
            new_val = self._value_from_mouse(mouse_x)
            if new_val != self.value:
                self.value = new_val
                if self.on_change:
                    self.on_change(self, self.value)

        # Release
        if self.is_dragging and not mouse_pressed:
            self.is_dragging = False
            if self.on_release:
                self.on_release(self)

        self.was_pressed_last_frame = mouse_pressed

        return self.value

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, dot):
        """Draw the slider using Dorothy

        Args:
            dot: Dorothy instance
        """
        track_h = max(4, self.height // 4)
        track_y = self.y + (self.height - track_h) // 2

        # Track background
        dot.fill(self.track_color)
        if self.border_width > 0:
            dot.stroke(self.border_color)
            dot.set_stroke_weight(self.border_width)
        else:
            dot.no_stroke()
        dot.rectangle((self.x, track_y), (self.x + self.width, track_y + track_h))

        # Track fill (left of handle)
        hx = self._handle_x()
        fill_right = hx + self.handle_width // 2
        if fill_right > self.x:
            dot.fill(self.track_fill_color)
            dot.no_stroke()
            dot.rectangle((self.x, track_y), (fill_right, track_y + track_h))

        # Handle
        if self.is_dragging:
            handle_color = self.handle_drag_color
        elif self.is_hovered:
            handle_color = self.handle_hover_color
        else:
            handle_color = self.handle_color

        dot.fill(handle_color)
        if self.border_width > 0:
            dot.stroke(self.border_color)
            dot.set_stroke_weight(self.border_width)
        else:
            dot.no_stroke()
        dot.rectangle((hx, self.y), (hx + self.handle_width, self.y + self.height))

        # Label
        if self.label:
            font_size = int(self.height * 0.4)
            dot.fill(self.label_color)
            dot.no_stroke()
            dot.text(self.label, self.x, self.y - font_size - 2, font_size)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def set_style(self, track_color=None, track_fill_color=None,
                  handle_color=None, handle_hover_color=None,
                  handle_drag_color=None, label_color=None,
                  border_color=None, border_width=None, handle_width=None):
        """Set slider visual style"""
        if track_color is not None: self.track_color = track_color
        if track_fill_color is not None: self.track_fill_color = track_fill_color
        if handle_color is not None: self.handle_color = handle_color
        if handle_hover_color is not None: self.handle_hover_color = handle_hover_color
        if handle_drag_color is not None: self.handle_drag_color = handle_drag_color
        if label_color is not None: self.label_color = label_color
        if border_color is not None: self.border_color = border_color
        if border_width is not None: self.border_width = border_width
        if handle_width is not None: self.handle_width = handle_width


# ---------------------------------------------------------------------------


class SliderManager:
    """Manages multiple sliders"""

    def __init__(self):
        self.sliders = []

    def add(self, slider):
        """Add a slider to the manager"""
        self.sliders.append(slider)
        return slider

    def create(self, x, y, width, height, min_val=0.0, max_val=1.0,
               value=None, label="", id=None,
               on_change=None, on_press=None, on_hover=None, on_release=None):
        """Create and add a slider"""
        slider = Slider(x, y, width, height, min_val, max_val, value, label, id,
                        on_change, on_press, on_hover, on_release)
        self.add(slider)
        return slider

    def update(self, mouse_x, mouse_y, mouse_pressed):
        """Update all sliders

        Returns:
            Dict mapping each slider to its current value
        """
        return {slider: slider.update(mouse_x, mouse_y, mouse_pressed)
                for slider in self.sliders}

    def draw(self, dot):
        """Draw all sliders"""
        for slider in self.sliders:
            slider.draw(dot)

    def get_by_id(self, id):
        """Get slider by ID"""
        for slider in self.sliders:
            if slider.id == id:
                return slider
        return None

    def remove(self, slider):
        """Remove a slider"""
        if slider in self.sliders:
            self.sliders.remove(slider)

    def remove_by_id(self, id):
        """Remove slider by ID"""
        slider = self.get_by_id(id)
        if slider:
            self.remove(slider)

    def clear(self):
        """Remove all sliders"""
        self.sliders.clear()

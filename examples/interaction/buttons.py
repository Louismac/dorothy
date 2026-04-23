"""GUI buttons with click and hover callbacks."""
from dorothy import Dorothy

dot = Dorothy()

class MySketch:

    def setup(self):
        dot.music.start_file_stream("../audio/gospel.wav")

        def on_click(btn):
            print(f"'{btn.text}' clicked")

        def on_hover(btn):
            print(f"'{btn.text}' hovered")

        dot.create_button(300, 250, 200, 50,
                          text="Click Me",
                          id="button1",
                          on_release=on_click,
                          on_hover=on_hover)

    def draw(self):
        dot.background((40, 40, 50))
        dot.update_buttons()
        dot.draw_buttons()

if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)

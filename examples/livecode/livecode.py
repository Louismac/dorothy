from dorothy import Dorothy
import importlib
import sketch

#RUN THIS FILE, THEN HAVE A FILE CALLED sketch.py in the same directory 
#with your MySketch class. Whenever you make changes to this and save, 
#it will be reflected in the open window withouth restarting

dot = Dorothy()
my_sketch = sketch.MySketch()
was_error = False

def setup_wrapper():
    global was_error
    try:
        importlib.reload(sketch)
        new_class = sketch.MySketch
        my_sketch.__class__ = new_class
        my_sketch.setup(dot)
        was_error = False
    except:
        if not was_error:
            print("error in setup, code not updated")
            was_error = True

def draw_wrapper():
    global was_error
    try:
        importlib.reload(sketch)
        new_class = sketch.MySketch
        my_sketch.__class__ = new_class
        my_sketch.draw(dot)
        was_error = False
    except:
        if not was_error:
            print("error in draw loop, code not updated")
            was_error = True

dot.start_loop(setup_wrapper, draw_wrapper)



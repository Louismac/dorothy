from dorothy import Dorothy

dot = Dorothy(width=1080, height=960, title="Dorothy Demo")
 

def setup():
    dot.background((255, 255, 255))

def draw():
    dot.circle((dot.mouse_x, dot.mouse_y), 20)

dot.start_loop(setup, draw)
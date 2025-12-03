from dorothy import Dorothy

dot = Dorothy()

class MySketch:
    def setup(self):
        self.col = (0,255,0)
        print("start")

    def run_once(self):
        print("run once")
        self.col = (0,0,255)
                
    def draw(self):
        dot.background(self.col)
        dot.fill(dot.red)
        dot.rectangle((0,dot.frames%40),(400,100))

#Start live code loop, updates whenever file
if __name__ == '__main__':
    import __main__
    dot.start_livecode_loop(__main__)
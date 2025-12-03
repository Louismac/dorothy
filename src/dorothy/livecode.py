from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import importlib
import traceback
import inspect
from pathlib import Path
import time
import sys
import runpy


class LiveCodeLoop:
    """Manages live code reloading for Dorothy sketches"""
    
    def __init__(self, dorothy_instance, sketch_module):
        self.dorothy = dorothy_instance
        self.sketch_module = sketch_module
        self.sketch_file = self._get_sketch_file()
        self.sketch_class = self._find_sketch_class()
        self.my_sketch = None
        self.was_error = False
        self.reload_requested = False
        self.reload_count = 0
        self.observer = None
        
        self._print_debug_info()
        self._initialize_sketch()
    
    def _get_sketch_file(self):
        """Get the file path to watch"""
        if self.sketch_module.__name__ == '__main__':
            return Path(sys.argv[0]).resolve()
        else:
            return Path(self.sketch_module.__file__)
    
    def _find_sketch_class(self):
        """Find the sketch class with setup and draw methods"""
        for name in dir(self.sketch_module):
            obj = getattr(self.sketch_module, name)
            if (inspect.isclass(obj) and 
                hasattr(obj, 'setup') and 
                hasattr(obj, 'draw') and
                obj.__module__ == self.sketch_module.__name__):
                return obj
        raise ValueError("No sketch class found with 'setup' and 'draw' methods")
    
    def _print_debug_info(self):
        """Print debug information"""
        print(f"🔍 DEBUG: Watching file: {self.sketch_file}")
        print(f"🔍 DEBUG: File exists: {self.sketch_file.exists()}")
        print(f"🔍 DEBUG: Watching directory: {self.sketch_file.parent}")
        print(f"🔍 DEBUG: Found sketch class: {self.sketch_class.__name__}")
    
    def _initialize_sketch(self):
        """Initialize the sketch instance with overridden __init__"""
        def new_init(self):
            print("Overridden init")
        
        self.sketch_class.__init__ = new_init
        self.my_sketch = self.sketch_class()
        print(self.my_sketch)
    
    def reload_sketch(self):
        """Reload the sketch without closing window"""
        try:
            print(f"📝 Reloading {self.sketch_file.name}...")
            self.reload_count += 1
            
            if self.sketch_module.__name__ == '__main__':
                file_globals = runpy.run_path(str(self.sketch_file), run_name='__live_reload__')
                new_class = self._find_sketch_class_in_globals(file_globals)
            else:
                if self.sketch_module.__name__ in sys.modules:
                    print(f"🔍 DEBUG: Removing {self.sketch_module.__name__} from sys.modules")
                importlib.reload(self.sketch_module)
                new_class = self._find_sketch_class()
            
            self.my_sketch.__class__ = new_class
            self.was_error = False
            
        except Exception:
            if not self.was_error:
                print("❌ Error reloading:")
                print(traceback.format_exc())
                self.was_error = True
    
    def _find_sketch_class_in_globals(self, globals_dict):
        """Find sketch class in a globals dictionary"""
        for name, obj in globals_dict.items():
            if (inspect.isclass(obj) and 
                hasattr(obj, 'setup') and 
                hasattr(obj, 'draw')):
                return obj
        raise ValueError("No sketch class found after reload")
    
    def setup_wrapper(self):
        """Initial setup"""
        print(f"🔍 DEBUG: setup_wrapper called")
        try:
            self.my_sketch.setup()
            self.was_error = False
        except Exception:
            if not self.was_error:
                print("❌ Error in setup:")
                print(traceback.format_exc())
                self.was_error = True
    
    def draw_wrapper(self):
        """Draw loop with reload checking"""
        if self.reload_requested:
            self.reload_requested = False
            print("self.my_sketch", self.my_sketch)
            self.reload_sketch()
            self._check_run_once_changed()
        
        try:
            self._handle_run_once()
            self.my_sketch.draw()
            self.was_error = False
        except Exception:
            if not self.was_error:
                print("❌ Error in draw:")
                print(traceback.format_exc())
                self.was_error = True
    
    def _check_run_once_changed(self):
        """Check if run_once method has changed"""
        if hasattr(self.my_sketch, 'run_once'):
            new_class = self.my_sketch.__class__
            func_key = inspect.getsource(new_class.run_once)
            
            if not hasattr(self.my_sketch, 'old_once_func'):
                self.my_sketch.old_once_func = func_key
            elif self.my_sketch.old_once_func != func_key:
                print(f"run_once changed after reload, will execute new version")
                self.my_sketch.once_ran = False
                self.my_sketch.old_once_func = func_key
    
    def _handle_run_once(self):
        """Handle run_once function execution"""
        if hasattr(self.my_sketch, 'run_once'):
            if not getattr(self.my_sketch, 'once_ran', False):
                self.my_sketch.run_once()
                print("running once")
                self.my_sketch.once_ran = True
    
    def start(self):
        """Start the live coding loop"""
        event_handler = SketchReloadHandler(self, self.sketch_file)
        self.observer = Observer()
        self.observer.schedule(event_handler, path=str(self.sketch_file.parent), recursive=False)
        self.observer.start()
        
        try:
            self.dorothy.start_loop(self.setup_wrapper, self.draw_wrapper)
        finally:
            print(f"🔍 DEBUG: Stopping observer...")
            self.observer.stop()
            self.observer.join()
            print(f"🔍 DEBUG: Observer stopped")


class SketchReloadHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self, livecode_loop, sketch_file):
        self.livecode_loop = livecode_loop
        self.sketch_file = sketch_file
        self.last_modified = 0
        self.event_count = 0
    
    def on_any_event(self, event):
        """Catch ALL events - modified, created, moved, etc."""
        self.event_count += 1
        
        if event.is_directory:
            print(f"   ⏭️  Skipping (directory)")
            return
        
        file_path = Path(event.src_path)
        
        if file_path.name == self.sketch_file.name and file_path.suffix == '.py':
            current_time = time.time()
            time_since_last = current_time - self.last_modified
            
            if time_since_last < 0.5:
                print(f"   ⏭️  DEBOUNCED (too soon, need 0.5s)")
                return
            
            self.last_modified = current_time
            self.livecode_loop.reload_requested = True
        else:
            print(f"   ⏭️  Not our target file")
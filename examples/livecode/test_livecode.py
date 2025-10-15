"""
Minimal test to diagnose live coding issues
Save this as test_livecode.py and run it
"""

from pathlib import Path
import time

# Test 1: File modification detection
print("=" * 60)
print("TEST 1: File modification time detection")
print("=" * 60)

sketch_file = Path("sketch.py")
print(f"Watching: {sketch_file.absolute()}")
print(f"File exists: {sketch_file.exists()}")

if sketch_file.exists():
    last_mtime = sketch_file.stat().st_mtime
    print(f"Current mtime: {last_mtime}")
    print(f"\nNow edit and SAVE sketch.py...")
    print("Watching for 10 seconds...\n")
    
    for i in range(20):  # Check for 10 seconds (20 * 0.5s)
        time.sleep(0.5)
        current_mtime = sketch_file.stat().st_mtime
        
        if current_mtime != last_mtime:
            print(f"✅ CHANGE DETECTED at {i*0.5:.1f}s!")
            print(f"   Old mtime: {last_mtime}")
            print(f"   New mtime: {current_mtime}")
            print(f"   Difference: {current_mtime - last_mtime:.3f}s")
            last_mtime = current_mtime
        
        # Print progress every 2 seconds
        if i % 4 == 0:
            print(f"   [{i*0.5:.1f}s] Still watching... (mtime: {current_mtime})")
    
    print("\n✅ Test 1 complete\n")
else:
    print("❌ sketch.py not found!\n")


# Test 2: Watchdog detection
print("=" * 60)
print("TEST 2: Watchdog file system events")
print("=" * 60)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    print("✅ Watchdog imported successfully")
    
    class TestHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            print(f"🔔 Event: {event.event_type} - {Path(event.src_path).name}")
    
    handler = TestHandler()
    observer = Observer()
    observer.schedule(handler, path=str(sketch_file.parent), recursive=False)
    observer.start()
    
    print(f"✅ Observer started")
    print(f"   Watching: {sketch_file.parent.absolute()}")
    print(f"   Observer alive: {observer.is_alive()}")
    print(f"\nNow edit and SAVE sketch.py...")
    print("Watching for 10 seconds...\n")
    
    time.sleep(10)
    
    observer.stop()
    observer.join()
    print("\n✅ Test 2 complete\n")
    
except ImportError:
    print("❌ Watchdog not installed")
    print("   Install with: pip install watchdog\n")
except Exception as e:
    print(f"❌ Error: {e}\n")


# Test 3: Check Dorothy integration
print("=" * 60)
print("TEST 3: Dorothy instance check")
print("=" * 60)

try:
    from dorothy import Dorothy
    import sketch
    
    print("✅ Imports successful")
    
    dot = Dorothy()
    print(f"✅ Dorothy created: {dot}")
    print(f"   Has start_livecode_loop: {hasattr(dot, 'start_livecode_loop')}")
    print(f"   Has start_livecode_loop_polling: {hasattr(dot, 'start_livecode_loop_polling')}")
    
    # Check if sketch module has MySketch
    print(f"   sketch.MySketch exists: {hasattr(sketch, 'MySketch')}")
    
    # Check Dorothy injection
    sketch.dot = dot
    print(f"   Injected dot into sketch module")
    
    # Try to create sketch instance
    try:
        my_sketch = sketch.MySketch(dot)
        print(f"✅ MySketch instance created: {my_sketch}")
    except TypeError:
        # Try without parameter
        my_sketch = sketch.MySketch()
        print(f"✅ MySketch instance created (no param): {my_sketch}")
    
    print("\n✅ Test 3 complete\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print()


print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print("\nIf Test 1 failed: Your editor might not be updating mtime")
print("If Test 2 failed: Watchdog isn't detecting events")
print("If Test 3 failed: Dorothy integration issue")
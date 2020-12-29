'''Imports'''
import tkinter as tk # Interface
from tkinter import ttk
from tkinter.filedialog import askopenfilename # File Dialog
from tkinter import font as tkfont # Fonts

import ntpath # Path Manipulation

from subprocess import Popen, PIPE # Running backend
from threading import Thread # Concurrency
from queue import Queue, Empty # Buffering output


'''Globals'''
WIDTH = 550 # Width of main window
HEIGHT = 300 # Height of main window


'''Main Tkinter App'''
class DriverApp(tk.Tk):
    def __init__(self, *args, **kwargs):        
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.FILENAME = tk.StringVar() # Name of model, eg. model.h5
        self.FILEPATH = tk.StringVar() # Full path to model
        self.SPEEDLIMIT = tk.IntVar() # Speed limit to be passed into backend
        
        self.title_font = tkfont.Font(family='Arial', size=25, weight="bold") # Font for "Driver" present on all frames
        
        # Container - a stack of frames
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        pages = (
                FrameLoadModel,
                FrameSetSpeed,
            ) # Tuple of frames in main window
        self.frames = {} # Dictionary of frames that will be populated with data from the tuple above
        for F in pages:
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew") # Placing all frames in the same location

        self.show_frame("FrameLoadModel") # Starting by showing the first frame in the pages tuple
    
    def show_frame(self, page_name):
        """Show a frame

        Args:
            page_name (String): Name of page
        """
        frame = self.frames[page_name]
        frame.tkraise() # Raise the currently shown frame to the top
 
'''Frames'''
class FrameLoadModel(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)        
        self.controller = controller
        
        label_title = tk.Label(self, text="Driver", font=controller.title_font)
        label_title.grid(row=0, column=0)
        
        def button_load_model_clicked():              
            filetypes = [
                ("Hierarchical Data binary files", '*.h5'), 
                ("All files", "*")
            ] # Validation - Ensuring the only files that may be picked are h5 files
            try:
                path = askopenfilename(filetypes=filetypes) # Show a file dialog window and return the path to the selected file
                _, tail = ntpath.split(path) # Sectioning off the last portion of the file path
                
                if path != "":
                    controller.FILEPATH.set(path) 
                    controller.FILENAME.set(tail)
                    controller.show_frame("FrameSetSpeed")
            except:
                controller.show_frame("FrameLoadModel")
        
        button_load_model = tk.Button(self, text="Load Model", height=2, width=10, command=button_load_model_clicked)
        button_load_model.grid(row=1, column=1, padx=WIDTH//4, pady=HEIGHT//4)

class FrameSetSpeed(tk.Frame):  
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label_title = tk.Label(self, text="Driver", font=controller.title_font)
        label_title.grid(row=0, column=0)
        
        button_back = tk.Button(self, text="Back", height=1, width=5, command=lambda: controller.show_frame("FrameLoadModel"))
        button_back.grid(row=1, column=0)
        
        label_loaded = tk.Label(self, text="Loaded Model:")
        label_loaded.grid(row=2, column=1)
        
        label_model_name = tk.Label(self, textvar=controller.FILENAME)
        label_model_name.grid(row=2, column=2)
        
        label_speed_limit = tk.Label(self, text="Speed Limit:")
        label_speed_limit.grid(row=3, column=1, pady=50)
        
        slider_speed_limit = tk.Scale(self, from_=1, to=30, resolution=0.1, orient="horizontal")
        slider_speed_limit.grid(row=3, column=2, columnspan=2, ipadx=50)
        
        def button_start_clicked():
            controller.SPEEDLIMIT.set(int(slider_speed_limit.get()))
            
            with open("data.txt", "w") as f:
                f.write(str(controller.FILEPATH.get()))
                f.write("\n")
                f.write(str(controller.SPEEDLIMIT.get()))
                f.close()
            
            drive(controller)

        button_start = tk.Button(self, text="Start", height=2, width=10, command=button_start_clicked)
        button_start.grid(row=4, column=2, padx=150)


'''Driving Process'''
def iter_except(function, exception):
    """Like iter() but stops on exception"""
    try:
        while True:
            yield function()
    except exception:
        return

class Driver:
    def __init__(self, root):
        self.root = root

        self.process = Popen(['python3', '-u', 'backend.py'], stdout=PIPE) # Start subprocess

        q = Queue(maxsize=1024)  # Limit output buffering (may stall subprocess)
        t = Thread(target=self.reader_thread, args=[q]) # Running separately to mainloop
        t.daemon = True # Close pipe if GUI process exits
        t.start()
        
        self.label_steering = tk.Label(root, text="Steering")
        self.label_steering.grid(row=0, column=0)
        
        self.label_steering_angle = tk.Label(root, text="")
        self.label_steering_angle.grid(row=1, column=0)
        
        self.label_throttle = tk.Label(root, text="Throttle")
        self.label_throttle.grid(row=0, column=1, padx=50)
        
        self.progressbar_throttle = ttk.Progressbar(root, orient="vertical", length=150)
        self.progressbar_throttle.grid(row=1, column=1, padx=50)
        
        self.update(q) # Begin update loop

    def reader_thread(self, q):
        """Read subprocess output and put it into the queue"""
        try:
            with self.process.stdout as pipe:
                for line in iter(pipe.readline, b''):
                    q.put(line)
        finally:
            q.put(None)

    def update(self, q):
        """Update GUI with items from the queue"""
        for line in iter_except(q.get_nowait, Empty):
            if line is None:
                self.quit()
                return
            else:
                stripped = line.decode().strip()
                vals = stripped.split(" ")
                try:
                    self.label_steering_angle["text"] = str(round(float(vals[0])*-180, 2))
                    self.progressbar_throttle["value"] = float(vals[1])*100
                except:
                    pass
                break # Update no more than once every 40ms
        self.root.after(40, self.update, q) # Schedule next update

    def quit(self):
        self.process.terminate() # Exit subprocess if GUI is closed
        self.root.destroy()

def drive(controller):
    window = tk.Toplevel(controller)
    
    driver = Driver(window)
    window.protocol("WM_DELETE_WINDOW", driver.quit)
    
    window.title("Monitor")
    window.geometry("550x300")
    

'''Program'''
if __name__ == "__main__":
    app = DriverApp()
    
    app.title("Driver")

    geometry = str(WIDTH) + "x" + str(HEIGHT)
    app.geometry(geometry)
    
    app.mainloop()
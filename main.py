from tkinter import ttk
from tkinter import *
import dataset as ds
import train as tr
import recognition_face as recFace
import time

class Recognition:
    def __init__(self,windows):
        self.wind = windows
        self.wind.title('Recognition Face')

        #Crear label en la pantalla
        frame = LabelFrame(self.wind, text='Ingrese el nombre de la persona')
        frame.grid(row=0, column=0, columnspan=3, pady=20)

        #inputs
        Label(frame, text='Nombre: ').grid(row=1, column=0)
        self.name = Entry(frame)
        self.name.focus()
        self.name.grid(row=1, column=1)

        #buton
        ttk.Button(frame, text='Iniciar captura', command = self.on_click).grid(row=2, columnspan=2, sticky= W + E)

    def on_click(self):
        print('into on_click in ' + self.name.get())
        ds.captureInfo(self.name.get())
        
def on_click_train():
    tr.train()    

def on_click_recognition():
    recFace.recognition_face()    
    

if __name__ == '__main__':
    windows = Tk()
    menuBar = Menu(windows)
    mnuFuntioncs=Menu(menuBar)
    mnuFuntioncs.add_command(label='1.Capture')
    mnuFuntioncs.add_command(label='2.Train', command=on_click_train)
    mnuFuntioncs.add_command(label='3.Recognition', command=on_click_recognition)
    mnuFuntioncs.add_separator()
    mnuFuntioncs.add_command(label='Exit')
    menuBar.add_cascade(label='Funtions', menu=mnuFuntioncs)
    windows.config(menu=menuBar)
    application = Recognition(windows)
    windows.mainloop()
    
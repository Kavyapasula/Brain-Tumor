import cv2
import numpy as np
import tkinter
from tkinter import *
from tkinter import filedialog, messagebox

# Initialize the Tkinter window
root = tkinter.Tk()
root.geometry('1100x600')
root.resizable(width=False, height=False)
root.title('Brain Tumor Detection')

# Load background image
try:
    filename = PhotoImage(file="bg.png")
    background_label = Label(root, image=filename)
    background_label.place(x=0, y=0)
except Exception as e:
    print("Background image not found:", e)

# Status bar
statusbar = Label(
    root, width=110, text="A project by Team SAHARA",
    font=("arial", 13, "bold"), bg="black", fg="white", relief=SUNKEN
)
statusbar.place(x=0, y=575)

# Global variable to store file path
files = [""]

# Function to select an image file
def get_files():
    file = filedialog.askopenfilename(title='Choose an Image File', filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file:
        files[0] = file  # Store the selected file path
        messagebox.showinfo('Success', 'Image has been selected')

# Function to close the application
def destroy():
    root.destroy()

# Canny Edge Detection function
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# Function to process the selected image
def inp():
    if not files[0]:
        messagebox.showerror("Error", "No image selected!")
        return

    path = files[0]
    org = cv2.imread(path)
    if org is None:
        messagebox.showerror("Error", "Invalid image file!")
        return

    dim = (500, 590)
    org = cv2.resize(org, dim)
    image = cv2.resize(cv2.imread(path), dim)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=14)
    closed = cv2.dilate(closed, None, iterations=13)

    ret, mask = cv2.threshold(closed, 155, 255, cv2.THRESH_BINARY)
    final = cv2.bitwise_and(image, image, mask=mask)

    canny = auto_canny(closed)
    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    
    final_output = np.concatenate((org, image), axis=1)
    cv2.imshow('Input V/S Output', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GUI Buttons
b1 = Button(root, padx=5, pady=5, width=12, bg='DodgerBlue1', fg='white', relief=GROOVE,
            command=get_files, text='SELECT IMG', font=('helvetica 15 bold'), activebackground='light green')
b1.place(x=780, y=150)

b2 = Button(root, padx=5, pady=5, width=12, bg='gray64', fg='white', relief=GROOVE,
            command=inp, text='SCAN', font=('helvetica 15 bold'), activebackground='light green')
b2.place(x=780, y=300)

b3 = Button(root, padx=5, pady=5, width=12, bg='orange red', fg='white', relief=GROOVE,
            text='EXIT', command=destroy, font=('helvetica 15 bold'), activebackground='red')
b3.place(x=780, y=450)

# Run the Tkinter main loop
root.mainloop()

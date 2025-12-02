import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from solver import *

# Replace this with the regular OpenCV method (no Google Colab)
classes = np.arange(0, 10)
model = load_model(r"C:\Users\VEDANT\Pictures\College\Assignment\Sem 5\CG\Sudoku\model-OCR.h5")
input_size = 48

def get_perspective(img, location, height=900, width=900):
    """Applies perspective transformation to extract Sudoku board."""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def get_InvPerspective(img, masked_num, location, height=900, width=900):
    """Inverse perspective transformation to map solved board back onto original image."""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result

def find_board(img):
    """Detects and extracts Sudoku board from the input image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location

def split_boxes(board):
    """Splits the Sudoku board into 81 individual cells."""
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0
            boxes.append(box)
    return boxes

def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays numbers on the Sudoku board."""
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:
                cv2.putText(img, str(numbers[(j * 9) + i]), (i * W + int(W / 2) - int((W / 4)), int((j + 0.7) * H)),
                            cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

def displayWords(img, numbers, words, color=(0, 0, 255)):
    """Displays words corresponding to Sudoku numbers."""
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:
                word = words[numbers[(j * 9) + i]]
                cv2.putText(img, word, (i * W + int(W / 4), int((j + 0.6) * H)), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2,
                            cv2.LINE_AA)
    return img


def process_image_for_sudoku(image_path):
    """Processes the image to extract the Sudoku puzzle and solve it."""
    img = cv2.imread(image_path)

    # Step 2: Detect and extract Sudoku board
    board, location = find_board(img)

    # Step 3: Process Sudoku board
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    # Get predictions
    prediction = model.predict(rois)
    predicted_numbers = [classes[np.argmax(p)] for p in prediction]
    board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

    # Solve the board
    solved_board_nums = get_board(board_num)

    # Display solved Sudoku in the cropped region
    solved_board = board.copy()
    displayNumbers(solved_board, solved_board_nums.flatten())

    # Overlay words on the solved cropped board
    word_board = board.copy()
    digit_to_word = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
    }
    displayWords(word_board, solved_board_nums.flatten(), digit_to_word, color=(0, 0, 255))

    # Map solved Sudoku back onto the original image
    binArr = np.where(np.array(predicted_numbers) > 0, 0, 1)
    flat_solved_board_nums = solved_board_nums.flatten() * binArr
    mask = np.zeros_like(board)
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
    inv = get_InvPerspective(img, solved_board_mask, location)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)

    return img, board, solved_board, word_board, combined


def display_image(image_path, label, description_label, description):
    """Displays an image in the given Tkinter label with a description."""
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk  # Keep reference to avoid garbage collection
    description_label.config(text=description)
    description_label.grid(row=description_label.grid_info()['row'], column=description_label.grid_info()['column'], padx=20, pady=5)

def upload_and_solve():
    """Handles file upload and Sudoku solving."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return

    try:
        # Process the image and get the solved Sudoku
        img, board, solved_board, word_board, combined = process_image_for_sudoku(file_path)

        # Save intermediate images
        cv2.imwrite("original_image.png", img)
        cv2.imwrite("cropped_sudoku.png", board)
        cv2.imwrite("solved_sudoku.png", solved_board)
        cv2.imwrite("solved_sudoku_with_words.png", word_board)
        cv2.imwrite("final_image.png", combined)

        # Display images with descriptions in GUI (show after upload)
        display_image("original_image.png", input_img_label, input_desc_label, "Original Image")
        display_image("cropped_sudoku.png", cropped_img_label, cropped_desc_label, "Cropped Sudoku Board")
        display_image("solved_sudoku.png", solved_img_label, solved_desc_label, "Solved Sudoku")
        display_image("solved_sudoku_with_words.png", word_img_label, word_desc_label, "Solved Sudoku with Words")
        display_image("final_image.png", final_img_label, final_desc_label, "Final Image with Solved Sudoku")

        messagebox.showinfo("Success", "Sudoku solved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to solve Sudoku: {e}")


# GUI Setup
root = tk.Tk()
root.title("Sudoku Solver")
root.geometry("1600x1000")  # Adjust window size to fit more images

# Title Label
title_label = tk.Label(root, text="Sudoku Solver", font=("Arial", 28, "bold"), pady=20)
title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=30)

# Upload Button
upload_button = tk.Button(root, text="Upload Sudoku Image", command=upload_and_solve, font=("Arial", 16), pady=10)
upload_button.grid(row=1, column=0, columnspan=4, pady=10)

# Image labels for displaying images
input_img_label = tk.Label(root)
input_img_label.grid(row=2, column=0, padx=20, pady=20)

cropped_img_label = tk.Label(root)
cropped_img_label.grid(row=2, column=1, padx=20, pady=20)

solved_img_label = tk.Label(root)
solved_img_label.grid(row=2, column=2, padx=20, pady=20)

word_img_label = tk.Label(root)
word_img_label.grid(row=2, column=3, padx=20, pady=20)

final_img_label = tk.Label(root)
final_img_label.grid(row=2, column=4, padx=20, pady=20)

# Description labels (initially hidden)
input_desc_label = tk.Label(root, font=("Arial", 12))
input_desc_label.grid(row=3, column=0)

cropped_desc_label = tk.Label(root, font=("Arial", 12))
cropped_desc_label.grid(row=3, column=1)

solved_desc_label = tk.Label(root, font=("Arial", 12))
solved_desc_label.grid(row=3, column=2)

word_desc_label = tk.Label(root, font=("Arial", 12))
word_desc_label.grid(row=3, column=3)

final_desc_label = tk.Label(root, font=("Arial", 12))
final_desc_label.grid(row=3, column=4)

# Start the GUI event loop
root.mainloop()
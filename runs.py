import sys
import cv2
import csv
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as PlatypusImage
from reportlab.lib import colors

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from io import BytesIO
import numpy as np
import math
import torch
from tracker import Tracker
object_labels = ['bike', 'bus', 'car', 'green-light', 'motorbike', 'red-light', 'truck', 'yellow-light']
limits = [300, 460, 1350, 560]
limits2 = [1020, 220, 1200, 380]
print(torch.cuda.is_available())

def remove_duplicates(arr):
    # Add the values along the second axis to create a new array of sums
    sums = np.sum(arr[:, :-3], axis=1)

    # Use np.unique to get the unique values of the sums array
    unique_sums, indices = np.unique(sums, return_index=True)

    # Use the indices to get the unique elements from the original array
    unique_arr = arr[indices]

    return unique_arr

def merge_arrays(arr1, arr2):
        if len(arr1) == 0 or len(arr2) ==0:
            return np.empty((0, 7), float)
        merged_arr = np.empty((0, 7), float)  # initialize empty array to store merged elements
        for i, (x1, x2, y1, y2, classname, score) in enumerate(arr1):
            distances = np.sqrt(np.sum((arr2[:, :4] - np.array([x1, x2, y1, y2]))**2, axis=1))  # calculate distances between arr1[i] and all elements in arr2
            closest_idx = np.argmin(distances)  # find the index of the closest element in arr2
            id = arr2[closest_idx, 4]  # get the id of the closest element in arr2
            merged_element = np.array([arr2[closest_idx, 0], arr2[closest_idx, 1], arr2[closest_idx, 2], arr2[closest_idx, 3], id, classname, score]).reshape(1, 7)  # create the merged element
            merged_arr = np.vstack((merged_arr, merged_element))  # add the merged element to the output array
        return remove_duplicates(merged_arr)

def ms_to_minutes_seconds(ms):
        total_seconds = int(ms / 1000)  # Convert milliseconds to seconds
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

def export_results_to_csv(table_output, file_path):
    # Check if there is data to export
    if not table_output:
        print("No data to export.")
        return

    # Define the CSV file header
    header = ["ID", "Type", "Timestamp"]

    # Open the CSV file in write mode
    with open(file_path, mode='w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)

        # Write the header to the CSV file
        csv_writer.writerow(header)

        # Write each row of data to the CSV file
        for row in table_output:
            csv_writer.writerow([row[0], row[1], row[2]])

    print("Data exported successfully to:", file_path)

def export_results_to_pdf(table_output, file_path):
    if not table_output:
        print("No data to export.")
        return

    # Create a new PDF document
    doc = SimpleDocTemplate(file_path, pagesize=letter)

    # Define the table data
    table_data = [
        ["ID", "Type", "Timestamp", "Image"]
    ]

    for row in table_output:
        table_data.append([str(row[0]), str(row[1]), str(row[2]), row[3]])

    # Create the table
    table = Table(table_data)

    # Add style to the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header background color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align for all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header padding
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Data background color
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),  # Data font
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),  # Data padding
    ])
    table.setStyle(style)

    # Build the PDF document with the table
    elements = [table]

    # Add the images to the PDF
    for row in table_output:
        image_qimage = row[3]
        image_pil = Image.fromqimage(image_qimage)

        # Convert PIL image to BytesIO
        image_bytesio = BytesIO()
        image_pil.save(image_bytesio, format="PNG")
        image_bytesio.seek(0)

        platypus_image = PlatypusImage(image_bytesio)
        platypus_image.drawHeight = 50  # Adjust height as needed
        platypus_image.drawWidth = 70   # Adjust width as needed
        platypus_image.hAlign = 'CENTER'
        elements.append(platypus_image)

    # Build the PDF document with the table and images
    doc.build(elements)

    print("Data exported successfully to:", file_path)


def remove_item_by_id(arr, item_id):
    return [item for item in arr if item[0] != item_id]

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the video frame index
        self.frame_index = 0
        self.is_paused = False
        self.is_Redlight = False
        self.model = self.load_model()
        self.tracker = Tracker()
        self.detections = []
        self.beforeTracking = []
        self.vehicleWhiteListed = []
        self.totalCount=[]
        self.violatedVehicle = []
        self.tableOutput=[]
        self.count = 0
        # Create the main layout
        layout = QVBoxLayout()
        # Create a label to display the video frames
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter) 
        layout.addWidget(self.label)

        # Create a button for selecting an MP4 file
        self.button = QPushButton("Select MP4 File", self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        # Create a button for exprot csv
        self.export_pdf_button = QPushButton("Export to PDF", self)
        self.export_pdf_button.clicked.connect(self.export_results_to_pdf)
        layout.addWidget(self.export_pdf_button)

        self.export_button = QPushButton("Export to CSV", self)
        self.export_button.clicked.connect(self.export_results_to_csv)
        layout.addWidget(self.export_button)

        # Create a button for pausing/resuming the video
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pause_button)

        # Create a button for clearing the video
        self.clear_button = QPushButton("Clear Video", self)
        self.clear_button.clicked.connect(self.clear_video)
        layout.addWidget(self.clear_button)

        # Create a label for loading message
        self.loading_label = QLabel(self)
        layout.addWidget(self.loading_label)

        self.table = QTableWidget(self)
        self.table.setColumnCount(4)  # Three columns: ID, Type, and Timestamp
        self.table.setHorizontalHeaderLabels(["IDs", "Type", "Timestamp", "Image"])
        layout.addWidget(self.table)
        self.table.setColumnWidth(3, 300)
        self.table.verticalHeader().setDefaultSectionSize(300)

        # Set the layout as the central widget
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set up a timer to update the video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def open_file_dialog(self):
        # Open a file dialog to select an MP4 file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select MP4 File", "", "MP4 Files (*.mp4)")

        if file_path:
            # Start the video player
            self.play_video(file_path)

    def export_results_to_csv(self):
        # Open a file dialog to select the destination CSV file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Export Data to CSV", "", "CSV Files (*.csv)")

        if file_path:
            export_results_to_csv(self.tableOutput, file_path)

    def export_results_to_pdf(self):
        # Open a file dialog to select the destination PDF file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Export Data to PDF", "", "PDF Files (*.pdf)")

        if file_path:
            export_results_to_pdf(self.tableOutput, file_path)
            
    def play_video(self, file_path):
        # Open the video file
        self.cap = cv2.VideoCapture(file_path)

        # Check if the video was opened successfully
        if not self.cap.isOpened():
            self.loading_label.setText("Failed to open video!")
            return

        # Start the timer to update the video frames
        self.timer.start(2)

        # Clear the loading message
        self.loading_label.clear()

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = YOLO("best.pt")
        model.fuse()
        model.to(device)

        return model
   
    def clear_table(self):
        self.table.clearContents()
        self.table.setRowCount(0)   

    def update_frame(self):
        currentFrameDetect = []
        currentFrameTracking = np.empty((0, 5), float)  # Initialize empty NumPy array with 5 columns
        if self.is_paused:
            return
        ret, frame = self.cap.read()

        if ret:
            # Convert the frame from BGR to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            # Run object detection on the frame
            results = self.model(frame)
            result = results[0]
            # Get the bounding box coordinates and class labels
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            labels = np.array(result.boxes.cls.cpu(), dtype="int")
            confs = np.array(result.boxes.conf.cpu(), dtype="float")

            self.detections.clear()  # Clear detections list before appending new detections

            # Draw bounding boxes and labels on the frame
            for bbox, label, conf in zip(bboxes, labels, confs):
                (x1, y1, x2, y2) = bbox
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                conf_format = math.ceil((conf * 100)) / 100

                # check for green and yellow light
                if (label == 3 or label == 8) and 5 not in labels:
                    self.is_Redlight = False

                # check for red light
                if label == 5 and 3 not in labels:
                    self.is_Redlight = True

                if self.is_Redlight:
                    # Draw a red line
                    cv2.line(frame_rgb, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)
                    cv2.line(frame_rgb, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (255, 0, 0), 2)
                if conf_format > 0.5:
                    if label == 0 or label == 2 or label == 4:
                        currentFrameDetect.append([x1, x2, y1, y2, label, conf])

                        self.detections.append([x1, y1, x2, y2, conf])

            # Convert lists to numpy arrays
            currentFrameDetect = np.array(currentFrameDetect)

            # Update the tracker with the current frame and detections
            self.tracker.update(frame, self.detections)
            for track in self.tracker.tracks:
                x1, y1, x2, y2 = track.bbox
                track_id = track.track_id
                currentFrameTracking = np.vstack((currentFrameTracking, [x1, x2, y1, y2, track_id]))

            finalTracker = merge_arrays(currentFrameDetect, currentFrameTracking)
            
            for track in finalTracker:
                x1 = int(track[0])
                y1 = int(track[2])
                x2 = int(track[1])
                y2 = int(track[3])
                id = int(track[4])
                label_Index = int(track[5])
                w, h = x2 - x1, y2 - y1 
                cx, cy = x1 + w//2, y1 + h//2

                cv2.circle(frame_rgb, (cx,cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame_rgb, f'Vehicle:{int(track[4])} - {object_labels[int(track[5])]}', (int(track[0]), int(track[2]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 225), 2)
                
                if(cy < limits[1] and self.vehicleWhiteListed.count(id) == 0):
                    self.vehicleWhiteListed.append(id)

                if(limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20 and self.is_Redlight == True and self.vehicleWhiteListed.count(id) == 0):
                    if self.totalCount.count(id) == 0:
                        vehicle_image = frame_rgb[y1:y2, x1:x2].copy()

                        # Convert the NumPy array to a QImage
                        height, width, channel = vehicle_image.shape
                        bytes_per_line = channel * width
                        q_image = QImage(vehicle_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

                        # Resize the QImage if necessary
                        q_image = q_image.scaled(250, 250, Qt.KeepAspectRatio)

                        self.totalCount.append(id)
                        self.tableOutput.append([id, object_labels[label_Index], ms_to_minutes_seconds(current_timestamp), q_image])
                        

                if(self.totalCount.count(id) > 0):
                    cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), (255,0,0), 2)
                    if(limits2[1] < cy < limits2[3] and cx > limits2[2] - 85):
                        self.totalCount.remove(id)
                        print(self.totalCount)
                        self.tableOutput = remove_item_by_id(self.tableOutput, id)
                        self.clear_table()  # Clear the QTableWidget after removing the item
                        cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), (0,255,0), 2)


                if self.tableOutput:
                    # Update the QTableWidget with the detected vehicle IDs
                    self.table.setRowCount(len(self.tableOutput))
                    for idx, violator in enumerate(self.tableOutput): 
                        item = QTableWidgetItem(str(violator[0]))
                        self.table.setItem(idx, 0, item)
                        item2 = QTableWidgetItem(str(violator[1]))
                        self.table.setItem(idx, 1, item2)
                        item3 = QTableWidgetItem(str(violator[2]))
                        self.table.setItem(idx, 2, item3)
                        image_item = QTableWidgetItem()
                        image_item.setData(Qt.DecorationRole, QPixmap.fromImage(violator[3]))
                        self.table.setItem(idx, 3, image_item)


            # Create a QImage from the frame
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(q_image)

            # Resize the pixmap if necessary
            pixmap = pixmap.scaled(854, 480, Qt.KeepAspectRatio)

            # Display the frame in the label
            self.label.setPixmap(pixmap)

        else:
            # The video has ended, stop the timer
            self.timer.stop()



    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText("Resume")
        else:
            self.pause_button.setText("Pause")

    def reset_video_state(self):
        # Reset all relevant variables and lists
        self.is_paused = False
        self.is_Redlight = False
        self.detections.clear()
        self.beforeTracking.clear()
        self.vehicleWhiteListed.clear()
        self.totalCount.clear()
        self.tableOutput.clear()
        self.table.setRowCount(0)
        self.count = 0

    def clear_video(self):
        # Clear the video and reset the video state
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.label.clear()
        self.reset_video_state()

    def closeEvent(self, event):
        # Release the video capture and close the application
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        event.accept()
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print("An error occurred:", e)
        app.quit()


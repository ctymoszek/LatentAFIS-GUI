# LatentAFIS-GUI
A graphical user interface (GUI) for the MSU Latent AFIS.

This interface was created using the Electron framework. It is designed for a Linux system and has been tested on Ubuntu 18.04.
Please note that this is not a production-quality application. It is designed for demonstration/visualization of results only.

![Screenshot of feature visualization](LatentAFIS-GUI1.jpg)
![Screenshot of search results](LatentAFIS-GUI2.jpg)

Setup:
1. Clone the repository to your machine.
2. Install the node package manager (npm) in the repository directory.
3. Use the command "npm start" to run the GUI.

## Data
Please note that this interface does not come with any fingerprint data (latent or rolled).
It will look for this data in the directories LatentAFIS-GUI/Python/Data/Latent and LatentAFIS-GUI/Python/Data/Rolled. These directories should contain both the image files and the corresponding template files generated by the MSU Latent AFIS (https://github.com/prip-lab/MSU-LatentAFIS). These should have the same base file name but different extension (i.e. "filename.bmp" and "filename.dat").
